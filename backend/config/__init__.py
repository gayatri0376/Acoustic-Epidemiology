try:
    import pymysql
    pymysql.install_as_MySQLdb()
    # Support MySQL 8.0+ on Django 6.1
    import django.db.backends.mysql.features as mysql_features
    mysql_features.DatabaseFeatures.minimum_database_version = (8, 0, 0)
except ImportError:
    pass
