from flask import Blueprint

blueprint = Blueprint(
    'admin_blueprint',
    __name__,
    url_prefix=''
)
