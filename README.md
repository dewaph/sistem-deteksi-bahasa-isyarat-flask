# [SISTEM DETEKSI BAHASA ISYARAT INDONESIA FLASK]()

This system is web-based and aims to make it easier for people to understand sign language. This system can translate sign language hand signals in real time to letters of the alphabet. This system was developed with the flask framework and also machine learning.

<br />

- ✅ `Database`: MySql
  - Silent fallback to `MySql`
- ✅ `DB Tools`: SQLAlchemy ORM, `Flask-Migrate`
- ✅ `Authentication`, Session Based, `OAuth` via **Github**
- ✅ Docker, `Flask-Minify` (page compression)
- 🚀 `Deployment`
  - `CI/CD` flow via `Render`, `Heroku`

<br />

![Web Deteksi Flask](<https://github.com/dewaph/images/blob/main/mockuper%20(1).png?raw=true>)

<br />

## ✨ Start the app in Docker

> 👉 **Step 1** - Download the code from the GH repository (using `GIT`)

```bash
$ git clone https://github.com/app-generator/flask-argon-dashboard.git
$ cd flask-argon-dashboard
```

<br />

> 👉 **Step 2** - Start the APP in `Docker`

```bash
$ docker-compose up --build
```

Visit `http://localhost:5085` in your browser. The app should be up & running.

<br />

## ✨ Manual Build

> Download the code

```bash
$ git clone https://github.com/app-generator/flask-argon-dashboard.git
$ cd flask-argon-dashboard
```

<br />

### 👉 Set Up for `Unix`, `MacOS`

> Install modules via `VENV`

```bash
$ virtualenv env
$ source env/bin/activate
$ pip3 install -r requirements.txt
```

<br />

> Set Up Flask Environment

```bash
$ export FLASK_APP=run.py
$ export FLASK_ENV=development
```

<br />

> Start the app

```bash
$ flask run
```

At this point, the app runs at `http://127.0.0.1:5000/`.

<br />

### 👉 Set Up for `Windows`

> Install modules via `VENV` (windows)

```
$ virtualenv env
$ .\env\Scripts\activate
$ pip3 install -r requirements.txt
```

<br />

> Set Up Flask Environment

```bash
$ # CMD
$ set FLASK_APP=run.py
$ set FLASK_ENV=development
$
$ # Powershell
$ $env:FLASK_APP = ".\run.py"
$ $env:FLASK_ENV = "development"
```

<br />

> Start the app

```bash
$ flask run
```

At this point, the app runs at `http://127.0.0.1:5000/`.

<br />

### 👉 Create Users

By default, the app redirects guest users to authenticate. In order to access the private pages, follow this set up:

- Start the app via `flask run`
- Access the `registration` page and create a new user:
  - `http://127.0.0.1:5000/register`
- Access the `sign in` page and authenticate
  - `http://127.0.0.1:5000/login`

<br />

## Recompile SCSS

To update the CSS, the recommended way is this:

- use a Node 16.x version
- navigate to `apps/static/assets`
- Install modules using `yarn` or `npm i`
- Edit SCSS files
- Recompile SCSS -> CSS via `gulp scss`
- Refresh the browser

<br />
