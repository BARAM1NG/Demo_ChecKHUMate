import streamlit as st
import streamlit_authenticator as stauth
from streamlit_authenticator.utilities.hasher import Hasher
import yaml


names = ["오구의모험","로그인테스트"]
usernames = ["오구", "로그인"]
passwords = ["1234","5678"] # yaml 파일 생성하고 비밀번호 지우기!


passwords_to_hash = ['abc', 'def']

hashed_passwords = Hasher(passwords_to_hash).generate() # 비밀번호 해싱

data = {
    "credentials" : {
        "usernames":{
            usernames[0]:{
                "name":names[0],
                "password":hashed_passwords[0]
                },
            usernames[1]:{
                "name":names[1],
                "password":hashed_passwords[1]
                }            
            }
    },
    "cookie": {
        "expiry_days" : 0, # 만료일, 재인증 기능 필요없으면 0으로 세팅
        "key": "some_signature_key",
        "name" : "some_cookie_name"
    },
    "preauthorized" : {
        "emails" : [
            "melsby@gmail.com"
        ]
    }
}

with open('config.yaml','w') as file:
    yaml.dump(data, file, default_flow_style=False)

with open('config.yaml', 'r') as file:
    config = yaml.safe_load(file)

## yaml 파일 데이터로 객체 생성
authenticator = stauth.Authenticate(
    config['credentials'],
    config['cookie']['name'],
    config['cookie']['key'],
    config['cookie']['expiry_days'],
    config['preauthorized']
)

## 로그인 위젯 렌더링
## log(in/out)(로그인 위젯 문구, 버튼 위치)
## 버튼 위치 = "main" or "sidebar"
name, authentication_status, username = authenticator.login("main")

# authentication_status : 인증 상태 (실패=>False, 값없음=>None, 성공=>True)
if authentication_status == False:
    st.error("Username/password is incorrect")

if authentication_status == None:
    st.warning("Please enter your username and password")

if authentication_status:
    authenticator.logout("Logout","sidebar")
    st.sidebar.title(f"Welcome {name}")
    
    ## 로그인 후 기능들 작성 ##