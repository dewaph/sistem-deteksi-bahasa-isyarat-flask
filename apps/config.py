import os, random, string

class Config(object):

    basedir = os.path.abspath(os.path.dirname(__file__))

    # Set up the App SECRET_KEY
    SECRET_KEY  = os.getenv('SECRET_KEY', None)
    if not SECRET_KEY:
        SECRET_KEY = ''.join(random.choice( string.ascii_lowercase  ) for i in range( 32 ))

    # Assets Management
    ASSETS_ROOT = os.getenv('ASSETS_ROOT', '/static/assets') 

    SQLALCHEMY_TRACK_MODIFICATIONS = False

    DB_ENGINE = 'mysql'
    DB_USERNAME = 'root'
    DB_PASS = ''
    DB_HOST = 'localhost'
    DB_PORT = '3306'
    DB_NAME = 'sistem_deteksi_isyarat'

    SQLALCHEMY_DATABASE_URI = '{}://{}:{}@{}:{}/{}'.format(
        DB_ENGINE,
        DB_USERNAME,
        DB_PASS,
        DB_HOST,
        DB_PORT,
        DB_NAME
    )        
      
    
class ProductionConfig(Config):
    DEBUG = False

    # Security
    SESSION_COOKIE_HTTPONLY = True
    REMEMBER_COOKIE_HTTPONLY = True
    REMEMBER_COOKIE_DURATION = 3600

class DebugConfig(Config):
    DEBUG = True


# Load all possible configurations
config_dict = {
    'Production': ProductionConfig,
    'Debug'     : DebugConfig
}
