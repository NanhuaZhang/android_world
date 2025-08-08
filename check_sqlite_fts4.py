import sqlite3

def check_fts4_support():
    try:
        conn = sqlite3.connect(":memory:")
        conn.execute("CREATE VIRTUAL TABLE test USING fts4(content TEXT);")
        print("✅ 当前 SQLite 支持 FTS4")
    except sqlite3.OperationalError as e:
        if "no such module: FTS4" in str(e):
            print("❌ 当前 SQLite 不支持 FTS4，需要更新或重新编译支持该模块")
        else:
            print(f"⚠️ 创建 FTS4 表失败，其他错误: {e}")
    finally:
        conn.close()

print(f"SQLite 版本: {sqlite3.sqlite_version}")
check_fts4_support()
