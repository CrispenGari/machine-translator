
class AppConfig:
    DEBUG = False
    TESTING = False
    ENV = "production"

class Development(AppConfig):
    DEBUG = True
    ENV = "development"

class Testing(AppConfig):
    TESTING = True

class Production(AppConfig):
    pass