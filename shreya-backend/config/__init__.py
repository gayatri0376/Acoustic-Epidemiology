# Use mysqlclient if installed; otherwise use PyMySQL as a drop-in replacement (easier on Windows).
# This only chooses a MySQL driver. It never falls back to a different database.
try:
    import MySQLdb  # noqa: F401
except ImportError:
    import pymysql

    pymysql.version_info = (2, 2, 1, 'final', 0)  # satisfies Django's driver version check
    pymysql.install_as_MySQLdb()
