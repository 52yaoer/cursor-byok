//! Verifies that cancelled transactions cannot leave a write lock in the pool.
use std::{
    future::Future,
    sync::Arc,
    task::{Context, Wake, Waker},
    time::Duration,
};

use cursor_server::store::Store;
use sqlx::{sqlite::SqliteConnectOptions, Connection, SqliteConnection};
use tokio::sync::Notify;

#[derive(Default)]
struct WorkerWake(Notify);

impl Wake for WorkerWake {
    fn wake(self: Arc<Self>) {
        self.0.notify_one();
    }

    fn wake_by_ref(self: &Arc<Self>) {
        self.0.notify_one();
    }
}

async fn setup() -> (tempfile::TempDir, Store, SqliteConnection) {
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("test.db");
    let store = Store::connect(&format!("sqlite://{}", path.display()))
        .await
        .unwrap();
    store.set_detailed_logging(true).await.unwrap();
    let observer = SqliteConnection::connect_with(
        &SqliteConnectOptions::new()
            .filename(path)
            .busy_timeout(Duration::from_millis(500)),
    )
    .await
    .unwrap();
    (directory, store, observer)
}

async fn assert_writes_continue(store: &Store, observer: &mut SqliteConnection) {
    sqlx::query("UPDATE service_settings SET value_json = 'false' WHERE setting_key = 'llm_detailed_logging'")
        .execute(&mut *observer)
        .await
        .expect("a cancelled request must release the SQLite write lock without restarting");
    assert!(!store.detailed_logging().await.unwrap());
    store.set_detailed_logging(true).await.unwrap();
    let stored: String = sqlx::query_scalar(
        "SELECT value_json FROM service_settings WHERE setting_key = 'llm_detailed_logging'",
    )
    .fetch_one(observer)
    .await
    .unwrap();
    assert_eq!(stored, "true", "later store writes must be committed");
}

#[tokio::test]
async fn cancelling_an_immediate_begin_releases_its_write_lock() {
    let (_directory, store, mut observer) = setup().await;
    let mut connection = store.pool().acquire().await.unwrap();
    // Keep BEGIN blocked until its first poll has registered the worker waker.
    let blocker = observer.begin_with("BEGIN IMMEDIATE").await.unwrap();
    let wake = Arc::new(WorkerWake::default());
    let waker = Waker::from(wake.clone());
    let mut context = Context::from_waker(&waker);
    let mut begin = Box::pin(connection.begin_with("BEGIN IMMEDIATE"));

    assert!(begin.as_mut().poll(&mut context).is_pending());
    blocker.rollback().await.unwrap();
    tokio::time::timeout(Duration::from_secs(2), wake.0.notified())
        .await
        .unwrap();
    // Acknowledge BEGIN, then cancel while SQLx is still validating its handle.
    assert!(begin.as_mut().poll(&mut context).is_pending());
    drop(begin);
    connection.ping().await.unwrap();
    assert!(connection.is_in_transaction());
    drop(connection);

    assert_writes_continue(&store, &mut observer).await;
}

#[tokio::test]
async fn cancelling_during_a_transaction_rolls_back_its_changes() {
    let (_directory, store, mut observer) = setup().await;
    let write_started = Arc::new(Notify::new());
    let writing_store = store.clone();
    let writing_started = write_started.clone();
    let task = tokio::spawn(async move {
        let mut transaction = writing_store
            .pool()
            .begin_with("BEGIN IMMEDIATE")
            .await
            .unwrap();
        sqlx::query("UPDATE service_settings SET value_json = 'false' WHERE setting_key = 'llm_detailed_logging'")
            .execute(&mut *transaction)
            .await
            .unwrap();
        writing_started.notify_one();
        std::future::pending::<()>().await;
        transaction.commit().await.unwrap();
    });
    write_started.notified().await;
    task.abort();
    assert!(task.await.unwrap_err().is_cancelled());

    assert!(store.detailed_logging().await.unwrap());
    assert_writes_continue(&store, &mut observer).await;
}

#[tokio::test]
async fn committed_and_rolled_back_connections_remain_usable() {
    let (_directory, store, mut observer) = setup().await;
    for commit in [true, false] {
        let mut transaction = store.pool().begin_with("BEGIN IMMEDIATE").await.unwrap();
        sqlx::query("UPDATE service_settings SET value_json = 'false' WHERE setting_key = 'llm_detailed_logging'")
            .execute(&mut *transaction)
            .await
            .unwrap();
        if commit {
            transaction.commit().await.unwrap();
        } else {
            transaction.rollback().await.unwrap();
        }
        assert_eq!(store.detailed_logging().await.unwrap(), !commit);
        assert_writes_continue(&store, &mut observer).await;
    }
}
