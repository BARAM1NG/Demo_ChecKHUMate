document.addEventListener("DOMContentLoaded", () => {
    const radios = document.querySelectorAll(".custom-radio input");

    radios.forEach(radio => {
        radio.addEventListener("change", () => {
            radios.forEach(r => {
                const label = r.closest('.custom-radio');
                if (r.checked) {
                    label.classList.add("selected");
                    label.classList.remove("deselected");
                } else {
                    label.classList.remove("selected");
                    label.classList.add("deselected");
                }
            });
        });
    });

    // 초기 상태 설정
    radios.forEach(r => {
        const label = r.closest('.custom-radio');
        if (r.checked) {
            label.classList.add("selected");
            label.classList.remove("deselected");
        } else {
            label.classList.remove("selected");
            label.classList.add("deselected");
        }
    });
});


