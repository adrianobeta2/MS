const track = document.querySelector(".slider-track");
const range = document.getElementById("sliderRange");
const thumbLower = document.getElementById("thumbLower");
const thumbUpper = document.getElementById("thumbUpper");
const outputLower = document.getElementById("outputLower");
const outputUpper = document.getElementById("outputUpper");

let min = 0, max = 255;
let lowerValue = 20, upperValue = 80;

// Atualiza posição inicial
function updatePositions() {
    const trackWidth = track.offsetWidth;
    const lowerPercent = ((lowerValue - min) / (max - min)) * 100;
    const upperPercent = ((upperValue - min) / (max - min)) * 100;

    thumbLower.style.left = `${lowerPercent}%`;
    thumbUpper.style.left = `${upperPercent}%`;
    range.style.left = `${lowerPercent}%`;
    range.style.width = `${upperPercent - lowerPercent}%`;

    outputLower.textContent = lowerValue;
    outputUpper.textContent = upperValue;
}

// Movimento do handle inferior
thumbLower.addEventListener("mousedown", () => {
    const moveLower = (e) => {
        const rect = track.getBoundingClientRect();
        const newValue = Math.min(
            Math.max(min, ((e.clientX - rect.left) / rect.width) * (max - min) + min),
            upperValue - 1
        );
        lowerValue = Math.round(newValue);
        updatePositions();
    };

    const stopLower = () => {
        document.removeEventListener("mousemove", moveLower);
        document.removeEventListener("mouseup", stopLower);
    };

    document.addEventListener("mousemove", moveLower);
    document.addEventListener("mouseup", stopLower);
});

// Movimento do handle superior
thumbUpper.addEventListener("mousedown", () => {
    const moveUpper = (e) => {
        const rect = track.getBoundingClientRect();
        const newValue = Math.max(
            Math.min(max, ((e.clientX - rect.left) / rect.width) * (max - min) + min),
            lowerValue + 1
        );
        upperValue = Math.round(newValue);
        updatePositions();
    };

    const stopUpper = () => {
        document.removeEventListener("mousemove", moveUpper);
        document.removeEventListener("mouseup", stopUpper);
    };

    document.addEventListener("mousemove", moveUpper);
    document.addEventListener("mouseup", stopUpper);
});

// Inicializa as posições
updatePositions();
