document.querySelector('form').addEventListener('submit', function(event) {
  var name = document.getElementById('name').value;
  if (!name) {
      alert('이름을 입력해주세요.');
      event.preventDefault(); // 폼 제출 중단
  }
});
