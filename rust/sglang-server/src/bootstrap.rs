//! PD KV bootstrap registry compatible with Python's
//! `CommonKVBootstrapServer`.

use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use axum::extract::{Query, State};
use axum::http::StatusCode;
use axum::response::{IntoResponse, Response};
use axum::routing::{get, post, put};
use axum::{Json, Router};

const ENTRY_CLEANUP_INTERVAL_ENV: &str = "SGLANG_DISAGGREGATION_BOOTSTRAP_ENTRY_CLEANUP_INTERVAL";
const ENTRY_CLEANUP_INTERVAL_DEFAULT_SECS: u64 = 120;

#[derive(Clone, serde::Serialize)]
struct RankInfo {
    rank_ip: String,
    rank_port: i64,
}

#[derive(serde::Serialize)]
struct PrefillServerInfo {
    attn_tp_size: i64,
    attn_cp_size: i64,
    dp_size: i64,
    pp_size: i64,
    page_size: Option<i64>,
    kv_cache_dtype: Option<String>,
    follow_bootstrap_room: bool,
    enable_dsa_cache_layer_split: bool,
    prefill_http_port: Option<i64>,
}

struct RoomEntry {
    dp_rank: i64,
    registered_at: Instant,
}

/// Mutable state copied from `CommonKVBootstrapServer`. The first rank
/// registration owns topology metadata, while every PUT increments readiness,
/// including a re-registration, matching the Python implementation.
#[derive(Default)]
struct Registry {
    attn_tp_size: Option<i64>,
    attn_cp_size: Option<i64>,
    dp_size: Option<i64>,
    pp_size: Option<i64>,
    page_size: Option<i64>,
    kv_cache_dtype: Option<String>,
    follow_bootstrap_room: Option<bool>,
    enable_dsa_cache_layer_split: Option<bool>,
    prefill_http_port: Option<i64>,
    prefill_ranks: HashMap<(i64, i64, i64, i64), RankInfo>,
    room_to_dp_rank: HashMap<i64, RoomEntry>,
    registered_count: i64,
}

impl Registry {
    fn expected_rank_count(&self) -> Option<i64> {
        Some(
            self.dp_size?
                .saturating_mul(self.attn_cp_size?)
                .saturating_mul(self.attn_tp_size?)
                .saturating_mul(self.pp_size?),
        )
    }

    fn is_ready(&self) -> bool {
        self.expected_rank_count()
            .is_some_and(|expected| self.registered_count >= expected)
    }

    fn remove_expired_rooms(&mut self, now: Instant, interval: Duration) {
        self.room_to_dp_rank
            .retain(|_, entry| now.saturating_duration_since(entry.registered_at) <= interval);
    }
}

type SharedRegistry = Arc<Mutex<Registry>>;

/// Integer fields on which Python explicitly calls `int(...)` accept either a
/// JSON integer or a numeric string, including surrounding whitespace.
#[derive(Clone, Copy)]
struct Int(i64);

impl<'de> serde::Deserialize<'de> for Int {
    fn deserialize<D: serde::Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        #[derive(serde::Deserialize)]
        #[serde(untagged)]
        enum Repr {
            Int(i64),
            String(String),
        }

        match Repr::deserialize(deserializer)? {
            Repr::Int(value) => Ok(Self(value)),
            Repr::String(value) => value
                .trim()
                .parse()
                .map(Self)
                .map_err(|_| serde::de::Error::custom(format!("invalid int: {value:?}"))),
        }
    }
}

#[derive(serde::Deserialize)]
struct RouteRegistration {
    attn_tp_size: i64,
    attn_tp_rank: i64,
    attn_cp_size: i64,
    attn_cp_rank: i64,
    attn_dp_size: i64,
    attn_dp_rank: i64,
    pp_size: i64,
    pp_rank: i64,
    system_dp_size: i64,
    system_dp_rank: i64,
    rank_ip: String,
    rank_port: Int,
    page_size: Int,
    kv_cache_dtype: Option<String>,
    #[serde(default)]
    prefill_http_port: Option<Int>,
    #[serde(default)]
    load_balance_method: Option<String>,
    #[serde(default)]
    enable_dsa_cache_layer_split: Option<bool>,
}

fn not_ready(registered_count: i64) -> Response {
    (
        StatusCode::SERVICE_UNAVAILABLE,
        format!("Prefill server not fully registered yet ({registered_count} workers registered)."),
    )
        .into_response()
}

async fn register_route(
    State(state): State<SharedRegistry>,
    Json(body): Json<RouteRegistration>,
) -> Response {
    let dp_size = if body.system_dp_size == 1 {
        body.attn_dp_size
    } else {
        body.system_dp_size
    };
    let dp_group = if body.system_dp_size == 1 {
        body.attn_dp_rank
    } else {
        body.system_dp_rank
    };

    let mut registry = state.lock().unwrap();
    registry.attn_tp_size.get_or_insert(body.attn_tp_size);
    registry.attn_cp_size.get_or_insert(body.attn_cp_size);
    registry.dp_size.get_or_insert(dp_size);
    registry.pp_size.get_or_insert(body.pp_size);
    registry.page_size.get_or_insert(body.page_size.0);
    if registry.kv_cache_dtype.is_none() {
        registry.kv_cache_dtype = body.kv_cache_dtype;
    }
    if registry.prefill_http_port.is_none() {
        registry.prefill_http_port = body.prefill_http_port.map(|port| port.0);
    }
    registry.follow_bootstrap_room.get_or_insert(
        body.load_balance_method
            .as_deref()
            .unwrap_or("follow_bootstrap_room")
            == "follow_bootstrap_room",
    );
    registry
        .enable_dsa_cache_layer_split
        .get_or_insert(body.enable_dsa_cache_layer_split.unwrap_or(false));
    registry.prefill_ranks.insert(
        (dp_group, body.attn_cp_rank, body.attn_tp_rank, body.pp_rank),
        RankInfo {
            rank_ip: body.rank_ip.clone(),
            rank_port: body.rank_port.0,
        },
    );
    registry.registered_count += 1;

    tracing::debug!(
        dp_group,
        cp_rank = body.attn_cp_rank,
        tp_rank = body.attn_tp_rank,
        pp_rank = body.pp_rank,
        rank_ip = %body.rank_ip,
        rank_port = body.rank_port.0,
        registered = registry.registered_count,
        expected = registry.expected_rank_count(),
        "registered prefill bootstrap rank"
    );
    "OK".into_response()
}

async fn query_route(
    State(state): State<SharedRegistry>,
    Query(query): Query<HashMap<String, String>>,
) -> Response {
    let rank = |name: &str| {
        query
            .get(name)
            .and_then(|value| value.trim().parse::<i64>().ok())
    };
    let (Some(dp_rank), Some(cp_rank), Some(tp_rank), Some(pp_rank)) = (
        rank("prefill_dp_rank"),
        rank("prefill_cp_rank"),
        rank("target_tp_rank"),
        rank("target_pp_rank"),
    ) else {
        return (
            StatusCode::BAD_REQUEST,
            "Missing inputs for bootstrap server.",
        )
            .into_response();
    };

    let registry = state.lock().unwrap();
    if !registry.is_ready() {
        return not_ready(registry.registered_count);
    }

    if (dp_rank, cp_rank, tp_rank, pp_rank) == (-1, -1, -1, -1) {
        return Json(PrefillServerInfo {
            attn_tp_size: registry.attn_tp_size.unwrap(),
            attn_cp_size: registry.attn_cp_size.unwrap(),
            dp_size: registry.dp_size.unwrap(),
            pp_size: registry.pp_size.unwrap(),
            page_size: registry.page_size,
            kv_cache_dtype: registry.kv_cache_dtype.clone(),
            follow_bootstrap_room: registry.follow_bootstrap_room.unwrap_or(true),
            enable_dsa_cache_layer_split: registry.enable_dsa_cache_layer_split.unwrap_or(false),
            prefill_http_port: registry.prefill_http_port,
        })
        .into_response();
    }

    match registry
        .prefill_ranks
        .get(&(dp_rank, cp_rank, tp_rank, pp_rank))
    {
        Some(info) => Json(info.clone()).into_response(),
        None => (
            StatusCode::NOT_FOUND,
            format!(
                "Bootstrap info not found for dp_rank={dp_rank} cp_rank={cp_rank} \
                 tp_rank={tp_rank} pp_rank={pp_rank}"
            ),
        )
            .into_response(),
    }
}

#[derive(serde::Deserialize)]
struct DpRankRegistration {
    bootstrap_room: Int,
    dp_rank: Int,
}

async fn register_dp_rank(
    State(state): State<SharedRegistry>,
    Json(body): Json<DpRankRegistration>,
) -> Response {
    state.lock().unwrap().room_to_dp_rank.insert(
        body.bootstrap_room.0,
        RoomEntry {
            dp_rank: body.dp_rank.0,
            registered_at: Instant::now(),
        },
    );
    "OK".into_response()
}

#[derive(serde::Deserialize)]
struct DpRankQuery {
    bootstrap_rooms: Vec<Int>,
}

async fn query_dp_ranks(
    State(state): State<SharedRegistry>,
    Json(body): Json<DpRankQuery>,
) -> Response {
    let registry = state.lock().unwrap();
    let result: HashMap<String, i64> = body
        .bootstrap_rooms
        .iter()
        .filter_map(|room| {
            let entry = registry.room_to_dp_rank.get(&room.0)?;
            Some((room.0.to_string(), entry.dp_rank))
        })
        .collect();
    Json(result).into_response()
}

fn router(state: SharedRegistry) -> Router {
    Router::new()
        .route("/route", put(register_route).get(query_route))
        .route("/register_dp_rank", post(register_dp_rank))
        .route("/query_dp_ranks", post(query_dp_ranks))
        .route("/health", get(|| async { "OK" }))
        .with_state(state)
}

async fn cleanup_expired_rooms(state: SharedRegistry, interval: Duration) {
    loop {
        tokio::time::sleep(interval).await;
        state
            .lock()
            .unwrap()
            .remove_expired_rooms(Instant::now(), interval);
    }
}

/// Running bootstrap server. Dropping the handle or calling `close` stops the
/// dedicated runtime and joins its native thread.
pub struct Handle {
    shutdown: Option<flume::Sender<()>>,
    thread: Option<std::thread::JoinHandle<()>>,
}

impl Handle {
    pub fn close(&mut self) {
        drop(self.shutdown.take());
        if let Some(thread) = self.thread.take() {
            let _ = thread.join();
        }
    }
}

impl Drop for Handle {
    fn drop(&mut self) {
        self.close();
    }
}

/// Start the bootstrap registry on its dedicated listener and runtime.
pub fn start(host: &str, port: u16) -> Result<Handle, String> {
    let listener = std::net::TcpListener::bind((host, port))
        .map_err(|error| format!("bootstrap bind {host}:{port} failed: {error}"))?;
    listener
        .set_nonblocking(true)
        .map_err(|error| format!("bootstrap listener set_nonblocking failed: {error}"))?;
    let local_addr = listener
        .local_addr()
        .map_err(|error| format!("bootstrap listener local_addr failed: {error}"))?;
    let runtime = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .map_err(|error| format!("build bootstrap runtime failed: {error}"))?;

    let cleanup_interval = Duration::from_secs(crate::environ::env_u64(
        ENTRY_CLEANUP_INTERVAL_ENV,
        ENTRY_CLEANUP_INTERVAL_DEFAULT_SECS,
    ));
    let (shutdown_tx, shutdown_rx) = flume::bounded::<()>(0);
    let thread = std::thread::Builder::new()
        .name("pd-bootstrap".into())
        .spawn(move || {
            runtime.block_on(async move {
                let state = SharedRegistry::default();
                tokio::spawn(cleanup_expired_rooms(state.clone(), cleanup_interval));
                let listener =
                    tokio::net::TcpListener::from_std(listener).expect("adopt bootstrap listener");
                tracing::info!(addr = %local_addr, "PD KV bootstrap server listening");
                tokio::select! {
                    result = axum::serve(listener, router(state)) => {
                        if let Err(error) = result {
                            tracing::error!(%error, "bootstrap server exited");
                        }
                    }
                    _ = shutdown_rx.recv_async() => {}
                }
            });
        })
        .map_err(|error| format!("spawn bootstrap thread failed: {error}"))?;

    Ok(Handle {
        shutdown: Some(shutdown_tx),
        thread: Some(thread),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::{Read, Write};
    use std::net::SocketAddr;

    const TOPOLOGY_QUERY: &str =
        "/route?prefill_dp_rank=-1&prefill_cp_rank=-1&target_tp_rank=-1&target_pp_rank=-1";

    fn request(
        addr: SocketAddr,
        method: &str,
        path: &str,
        body: Option<&serde_json::Value>,
    ) -> (u16, String) {
        let body = body.map(ToString::to_string).unwrap_or_default();
        let mut connection = std::net::TcpStream::connect(addr).expect("connect");
        let request = format!(
            "{method} {path} HTTP/1.1\r\nHost: test\r\nContent-Type: application/json\r\n\
             Content-Length: {}\r\nConnection: close\r\n\r\n{body}",
            body.len()
        );
        connection.write_all(request.as_bytes()).unwrap();

        let mut response = String::new();
        connection.read_to_string(&mut response).unwrap();
        let status = response
            .split_whitespace()
            .nth(1)
            .expect("status")
            .parse()
            .expect("numeric status");
        let body = response
            .split_once("\r\n\r\n")
            .map(|(_, body)| body.to_string())
            .unwrap_or_default();
        (status, body)
    }

    fn route_registration(overrides: serde_json::Value) -> serde_json::Value {
        let mut body = serde_json::json!({
            "attn_tp_size": 1,
            "attn_tp_rank": 0,
            "attn_cp_size": 1,
            "attn_cp_rank": 0,
            "attn_dp_size": 1,
            "attn_dp_rank": 0,
            "pp_size": 1,
            "pp_rank": 0,
            "system_dp_size": 1,
            "system_dp_rank": 0,
            "rank_ip": "10.0.0.1",
            "rank_port": 17000,
            "page_size": 64,
            "kv_cache_dtype": "auto",
            "load_balance_method": "follow_bootstrap_room",
            "enable_dsa_cache_layer_split": false,
            "prefill_http_port": 30000
        });
        body.as_object_mut()
            .unwrap()
            .extend(overrides.as_object().unwrap().clone());
        body
    }

    fn start_on_free_port() -> (Handle, SocketAddr) {
        let probe = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
        let addr = probe.local_addr().unwrap();
        drop(probe);
        (start("127.0.0.1", addr.port()).unwrap(), addr)
    }

    #[test]
    fn route_contract_matches_python_client() {
        let (_handle, addr) = start_on_free_port();

        assert_eq!(request(addr, "GET", TOPOLOGY_QUERY, None).0, 503);
        assert_eq!(
            request(addr, "GET", "/route?prefill_dp_rank=0", None).0,
            400
        );

        let registration = route_registration(serde_json::json!({
            "rank_port": "17000",
            "page_size": "64",
            "prefill_http_port": "30000"
        }));
        assert_eq!(
            request(addr, "PUT", "/route", Some(&registration)),
            (200, "OK".into())
        );

        let (status, body) = request(addr, "GET", TOPOLOGY_QUERY, None);
        assert_eq!(status, 200);
        assert_eq!(
            serde_json::from_str::<serde_json::Value>(&body).unwrap(),
            serde_json::json!({
                "attn_tp_size": 1,
                "attn_cp_size": 1,
                "dp_size": 1,
                "pp_size": 1,
                "page_size": 64,
                "kv_cache_dtype": "auto",
                "follow_bootstrap_room": true,
                "enable_dsa_cache_layer_split": false,
                "prefill_http_port": 30000
            })
        );

        let (status, body) = request(
            addr,
            "GET",
            "/route?prefill_dp_rank=0&prefill_cp_rank=0&target_tp_rank=0&target_pp_rank=0",
            None,
        );
        assert_eq!(status, 200);
        assert_eq!(
            serde_json::from_str::<serde_json::Value>(&body).unwrap(),
            serde_json::json!({"rank_ip": "10.0.0.1", "rank_port": 17000})
        );

        assert_eq!(
            request(
                addr,
                "GET",
                "/route?prefill_dp_rank=1&prefill_cp_rank=0&target_tp_rank=0&target_pp_rank=0",
                None,
            )
            .0,
            404
        );
        assert_eq!(request(addr, "POST", "/route", None).0, 405);
    }

    #[test]
    fn system_dp_controls_readiness_and_rank_lookup() {
        let (_handle, addr) = start_on_free_port();

        for (rank, ip) in [(0, "10.0.0.1"), (1, "10.0.0.2")] {
            let registration = route_registration(serde_json::json!({
                "system_dp_size": 2,
                "system_dp_rank": rank,
                "rank_ip": ip
            }));
            assert_eq!(request(addr, "PUT", "/route", Some(&registration)).0, 200);
            if rank == 0 {
                assert_eq!(request(addr, "GET", TOPOLOGY_QUERY, None).0, 503);
            }
        }

        let (status, body) = request(addr, "GET", TOPOLOGY_QUERY, None);
        assert_eq!(status, 200);
        assert_eq!(
            serde_json::from_str::<serde_json::Value>(&body).unwrap()["dp_size"],
            2
        );

        let (status, body) = request(
            addr,
            "GET",
            "/route?prefill_dp_rank=1&prefill_cp_rank=0&target_tp_rank=0&target_pp_rank=0",
            None,
        );
        assert_eq!(status, 200);
        assert_eq!(
            serde_json::from_str::<serde_json::Value>(&body).unwrap()["rank_ip"],
            "10.0.0.2"
        );
    }

    #[test]
    fn dp_room_round_trip_and_health() {
        let (_handle, addr) = start_on_free_port();

        assert_eq!(request(addr, "GET", "/health", None), (200, "OK".into()));
        assert_eq!(
            request(
                addr,
                "POST",
                "/register_dp_rank",
                Some(&serde_json::json!({"bootstrap_room": "42", "dp_rank": "3"})),
            ),
            (200, "OK".into())
        );

        let (status, body) = request(
            addr,
            "POST",
            "/query_dp_ranks",
            Some(&serde_json::json!({"bootstrap_rooms": ["42", 99]})),
        );
        assert_eq!(status, 200);
        assert_eq!(
            serde_json::from_str::<serde_json::Value>(&body).unwrap(),
            serde_json::json!({"42": 3})
        );
    }

    #[test]
    fn expired_rooms_are_removed_without_touching_fresh_rooms() {
        let now = Instant::now();
        let interval = Duration::from_secs(10);
        let mut registry = Registry::default();
        registry.room_to_dp_rank.insert(
            1,
            RoomEntry {
                dp_rank: 0,
                registered_at: now - Duration::from_secs(11),
            },
        );
        registry.room_to_dp_rank.insert(
            2,
            RoomEntry {
                dp_rank: 1,
                registered_at: now - Duration::from_secs(10),
            },
        );

        registry.remove_expired_rooms(now, interval);

        assert!(!registry.room_to_dp_rank.contains_key(&1));
        assert_eq!(
            registry.room_to_dp_rank.get(&2).map(|entry| entry.dp_rank),
            Some(1)
        );
    }

    #[test]
    fn handle_owns_listener_and_close_releases_it() {
        let (mut handle, addr) = start_on_free_port();
        assert!(
            start("127.0.0.1", addr.port()).is_err(),
            "a second server cannot bind the live listener"
        );

        handle.close();
        let listener = std::net::TcpListener::bind(addr)
            .expect("close must stop the thread and release the listener");
        drop(listener);
    }
}
