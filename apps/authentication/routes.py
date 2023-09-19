from flask import render_template, redirect, request, url_for
from flask_login import (
    current_user,
    login_user,
    logout_user
)

from apps import db, login_manager
from apps.authentication import blueprint
from apps.authentication.forms import LoginForm, CreateAccountForm
from apps.authentication.models import Users


@blueprint.route('/')
def route_default():
    if current_user.is_authenticated:
        if current_user.is_admin():
            return redirect(url_for('admin_blueprint.admin'))
        return redirect(url_for('home_blueprint.beranda'))  
    return redirect(url_for('authentication_blueprint.login'))

# Login & Registration

@blueprint.route('/login', methods=['GET', 'POST'])
def login():
    login_form = LoginForm(request.form)
    if 'login' in request.form:
        username = request.form['username']
        password = request.form['password']
        user = Users.query.filter_by(username=username).first()
        if user and user.password == password:
            login_user(user)
            return redirect(url_for('authentication_blueprint.route_default'))
        return render_template('accounts/login.html',
                               msg='Username atau Password salah',
                               form=login_form)
    if not current_user.is_authenticated:
        return render_template('accounts/login.html',
                               form=login_form)


@blueprint.route('/register', methods=['GET', 'POST'])
def register():
    create_account_form = CreateAccountForm(request.form)
    if 'register' in request.form:
        username = request.form['username']
        user = Users.query.filter_by(username=username).first()
        if user:
            return render_template('accounts/register.html',
                                   msg='Username sudah ada!',
                                   success=False,
                                   form=create_account_form)
        user = Users(**request.form)
        db.session.add(user)
        db.session.commit()
        logout_user()
        return render_template('accounts/register.html',
                               msg='Akun berhasil terdaftar',
                               success=True,
                               form=create_account_form)
    else:
        return render_template('accounts/register.html', form=create_account_form)


@blueprint.route('/logout')
def logout():
    logout_user()
    return redirect(url_for('authentication_blueprint.login')) 

# Errors

@login_manager.unauthorized_handler
def unauthorized_handler():
    return render_template('home/page-403.html'), 403


@blueprint.errorhandler(403)
def access_forbidden(error):
    return render_template('home/page-403.html'), 403


@blueprint.errorhandler(404)
def not_found_error(error):
    return render_template('home/page-404.html'), 404


@blueprint.errorhandler(500)
def internal_error(error):
    return render_template('home/page-500.html'), 500
