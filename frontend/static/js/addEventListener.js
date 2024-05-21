document.addEventListener("DOMContentLoaded", function() {
    // dropdown-menu 내의 모든 a 태그 선택
    const links = document.querySelectorAll('.dropdown-menu a');

    // 각 a 태그에 클릭 이벤트 리스너 추가
    links.forEach(link => {
        link.addEventListener('click', function(e) {
            e.preventDefault(); // 기본 동작 방지
            // 이미 체크되어 있다면 체크 제거
            if(this.classList.contains('checked')) {
                this.classList.remove('checked');
            } else {
                // 체크되지 않았다면 체크 추가
                this.classList.add('checked');
            }
        });
    });
});