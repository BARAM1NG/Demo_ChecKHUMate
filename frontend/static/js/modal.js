// modal.js
let modal = document.getElementById("filterModal");
// 클래스 이름으로 버튼 선택
let btn = document.querySelector(".category-button-a");
let span = document.getElementsByClassName("close")[0];

btn.onclick = function() {
    modal.style.display = "block";
}

span.onclick = function() {
    modal.style.display = "none";
}

window.onclick = function(event) {
    if (event.target == modal) {
        modal.style.display = "none";
    }
}

document.getElementById("filterForm").onsubmit = function(event) {
    event.preventDefault();
    // 필터링 로직 구현
}
