// modal.js
let modal = document.getElementById("filterModal");
let btn = document.getElementById("filterBtn");
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
    // 예: 입력된 값을 가져와서 해당 조건에 맞는 룸메이트 정보를 조회하는 로직
}
