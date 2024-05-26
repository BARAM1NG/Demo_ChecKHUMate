from flask import Flask, render_template

# 'template_folder' 파라미터를 추가하여 HTML 파일 경로를 지정
app = Flask(__name__, template_folder='../../frontend/templates', static_folder='../../frontend/static')

@app.route("/")
def main():
    # 'main.html' 파일을 렌더링
    return render_template("main.html")

@app.route("/my_profile.html")
def my_profile():
    # 'main.html' 파일을 렌더링
    return render_template("my_profile.html")

@app.route("/check.html")
def check():
    # 'my_profile.html' 파일을 렌더링
    return render_template("check.html")

@app.route("/my_page_2.html")
def my_page_2():
    # 'main.html' 파일을 렌더링
    return render_template("my_page_2.html")

@app.route("/roommate_profile.html")
def roommate_profile():
    # 'main.html' 파일을 렌더링
    return render_template("roommate_profile.html")


if __name__ == "__main__":
    app.run(debug=True)