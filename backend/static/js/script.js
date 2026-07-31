// ======================================================
// Get HTML Elements
// ======================================================
const cameraBtn = document.getElementById("cameraBtn");

const cameraModal = document.getElementById("cameraModal");

const video = document.getElementById("video");

const canvas = document.getElementById("canvas");

const captureBtn = document.getElementById("captureBtn");

const closeCameraBtn = document.getElementById("closeCameraBtn");

let stream = null;


const imageInput = document.getElementById("imageInput");
const preview = document.getElementById("preview");

const predictBtn = document.getElementById("predictBtn");

const loading = document.getElementById("loading");

const result = document.getElementById("result");

const crop = document.getElementById("crop");
const disease = document.getElementById("disease");
const confidence = document.getElementById("confidence");
const confidenceBar = document.getElementById("confidenceBar");

const description = document.getElementById("description");
const cause = document.getElementById("cause");
const symptoms = document.getElementById("symptoms");
const treatment = document.getElementById("treatment");
const prevention = document.getElementById("prevention");
const severity = document.getElementById("severity");
const confidenceMessage = document.getElementById("confidenceMessage");

const topPredictions = document.getElementById("topPredictions");

const warningBox = document.getElementById("warningBox");
const resetBtn = document.getElementById("resetBtn");

// ======================================================
// Preview Image
// ======================================================

imageInput.addEventListener("change", function () {

    const file = this.files[0];

    if (!file) {

        preview.src = "";
        preview.style.display = "none";
        return;

    }

    const reader = new FileReader();

    reader.onload = function (e) {

        preview.src = e.target.result;
        preview.style.display = "block";

    };

    reader.readAsDataURL(file);

});

// ======================================================
// Helper Function
// ======================================================

function createList(element, items) {

    element.innerHTML = "";

    if (!items || !Array.isArray(items)) {
        return;
    }

    items.forEach(item => {

        const li = document.createElement("li");

        li.textContent = item;

        element.appendChild(li);

    });

}
function showTopPredictions(predictions) {

    topPredictions.innerHTML = "";
    topPredictions.style.display = "none";

    predictions.forEach((item, index) => {

        const div = document.createElement("div");

        div.className = "top-card";

        let medal = "";

        if(index === 0) medal = "🥇";
        else if(index === 1) medal = "🥈";
        else medal = "🥉";

        div.innerHTML = `
            <h4>${medal} ${item.crop}</h4>
            <p>${item.disease}</p>
            <strong>${item.confidence}%</strong>
        `;

        topPredictions.appendChild(div);

    });

}

// ======================================================
// Predict Button
// ======================================================

predictBtn.addEventListener("click", async function () {

    const file = imageInput.files[0];

    if (!file) {

        alert("Please upload a leaf image.");
        return;

    }

    loading.classList.remove("hidden");
    result.classList.add("hidden");

    predictBtn.disabled = true;
    predictBtn.innerHTML = "Predicting...";

    const formData = new FormData();

    formData.append("image", file);

    try {

        const response = await fetch("/predict", {

            method: "POST",

            body: formData

        });
        console.log("Response status:", response.status);


        const data = await response.json();
        console.log("Received data:", data);
        console.log(data);

        loading.classList.add("hidden");
        resetBtn.classList.remove("hidden");

        predictBtn.disabled = false;
        predictBtn.innerHTML = "🔍 Predict Disease";

        if (data.success === false)  {

    loading.classList.add("hidden");

    predictBtn.disabled = false;

    predictBtn.innerHTML = "🔍 Predict Disease";

  console.log("Before:", result.className);

result.classList.remove("hidden");

// Force it to show
result.style.display = "block";

console.log("After:", result.className);
console.log("Result element:", result);

    crop.textContent = "-";
    disease.textContent = "Not a Plant Leaf";
    confidence.textContent = data.leaf_confidence + "%";
    confidenceBar.style.width = data.leaf_confidence + "%";
    confidenceBar.textContent = data.leaf_confidence + "%";

    description.textContent = data.message;
    cause.textContent = "-";

    symptoms.innerHTML = "";
    treatment.innerHTML = "";
    prevention.innerHTML = "";

    severity.textContent = "Invalid Image";
    severity.className = "severity high";

    confidenceMessage.innerHTML =
        "Please upload a clear image containing a single plant leaf.";

    warningBox.style.display = "block";
    warningBox.innerHTML = `
        <strong>Accepted Images:</strong>
        <ul>
            <li>🌿 One plant leaf</li>
            <li>📷 Good lighting</li>
            <li>🎯 Leaf occupies most of the image</li>
        </ul>
    `;

    return;
}
        // ============================================
        // Basic Prediction
        // ============================================

        crop.textContent = data.crop;
        console.log(crop);
        console.log("Crop updated");

        disease.textContent = data.disease;
        console.log("Disease updated");



        confidence.textContent = data.confidence + "%";
        console.log("Confidence updated");


        confidenceBar.style.width = data.confidence + "%";


        confidenceBar.textContent = data.confidence + "%";
        if (data.confidence >= 90) {

    confidenceMessage.innerHTML =
        "✅ Excellent confidence. The prediction is highly reliable.";

    warningBox.style.display = "none";

}

else if (data.confidence >= 70) {

    confidenceMessage.innerHTML =
        "🟡 Good confidence. Verify the symptoms before treatment.";

    warningBox.style.display = "none";

}

else {

    confidenceMessage.innerHTML =
        "⚠ Low confidence prediction.";

    warningBox.style.display = "block";

    warningBox.innerHTML = `
        <strong>Recommendation:</strong>
        <ul>
            <li>Upload one leaf only.</li>
            <li>Use good lighting.</li>
            <li>Avoid blurry images.</li>
            <li>Capture the infected area clearly.</li>
        </ul>
    `;

}
topPredictions.style.display = "grid";
showTopPredictions(data.top_predictions);
console.log("Top predictions updated");

        // ============================================
        // Disease Information
        // ============================================

        description.textContent = data.description;
        console.log("Description updated");

        cause.textContent = data.cause;

        createList(symptoms, data.symptoms);

        createList(treatment, data.treatment);

        createList(prevention, data.prevention);

        severity.textContent = data.severity;

        // ============================================
        // Severity Badge Color
        // ============================================

        severity.className = "severity";

        if (data.severity === "Healthy") {

            severity.classList.add("healthy");

        }

        else if (data.severity === "Medium") {

            severity.classList.add("medium");

        }

        else {

            severity.classList.add("high");

        }

        result.classList.remove("hidden");
        console.log(result);
        console.log("RESULT SHOWN");

    }

catch (error) {

    console.error("ERROR:", error);

    alert(error.message);

}
});
resetBtn.addEventListener("click", () => {

    imageInput.value = "";

    preview.src = "";

    preview.style.display = "none";

    result.classList.add("hidden");

    resetBtn.classList.add("hidden");

    confidenceBar.style.width = "0%";

    confidenceBar.textContent = "0%";

    confidenceMessage.innerHTML = "";

    warningBox.style.display = "none";
    topPredictions.innerHTML = "";
    topPredictions.style.display = "none";

});

cameraBtn.addEventListener("click", async () => {

    try{

        stream = await navigator.mediaDevices.getUserMedia({

            video:true

        });

        video.srcObject = stream;

        cameraModal.classList.remove("hidden");

    }

    catch(err){

        alert("Unable to access camera.");

        console.error(err);

    }

});
captureBtn.addEventListener("click", () => {

    // Set canvas size equal to video frame
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;

    // Draw current video frame onto canvas
    const ctx = canvas.getContext("2d");
    ctx.drawImage(video, 0, 0, canvas.width, canvas.height);

    // Convert canvas to image file
    canvas.toBlob((blob) => {

        const file = new File([blob], "captured_leaf.jpg", {
            type: "image/jpeg"
        });

        // Put captured image into file input
        const dataTransfer = new DataTransfer();
        dataTransfer.items.add(file);
        imageInput.files = dataTransfer.files;

        // Show preview
        preview.src = URL.createObjectURL(file);
        preview.style.display = "block";

        // Stop camera
        if (stream) {
            stream.getTracks().forEach(track => track.stop());
        }

        // Close modal
        cameraModal.classList.add("hidden");

    }, "image/jpeg");

});
closeCameraBtn.addEventListener("click", () => {

    if(stream){

        stream.getTracks().forEach(track=>track.stop());

    }

    cameraModal.classList.add("hidden");

});
// ======================================================
// Predict Disease
// ======================================================

predictBtn.addEventListener("click", async () => {

    const file = imageInput.files[0];

    if (!file) {
        alert("Please upload a leaf image.");
        return;
    }

    loading.classList.remove("hidden");
    result.classList.add("hidden");

    predictBtn.disabled = true;
    predictBtn.innerHTML = "Predicting...";

    const formData = new FormData();
    formData.append("image", file);

    try {

        const response = await fetch("/predict", {
            method: "POST",
            body: formData
        });

        const data = await response.json();

        console.log("Response:", data);

        loading.classList.add("hidden");

        predictBtn.disabled = false;
        predictBtn.innerHTML = "🔍 Predict Disease";

        // =====================================
        // Invalid Image / Not Leaf
        // =====================================

        if (data.success === false) {

            result.classList.remove("hidden");

            crop.textContent = "-";
            disease.textContent = "Not a Plant Leaf";

            confidence.textContent =
                (data.leaf_confidence ?? 0) + "%";

            confidenceBar.style.width =
                (data.leaf_confidence ?? 0) + "%";

            confidenceBar.textContent =
                (data.leaf_confidence ?? 0) + "%";

            description.textContent = data.message;

            cause.textContent = "-";

            symptoms.innerHTML = "";
            treatment.innerHTML = "";
            prevention.innerHTML = "";

            topPredictions.innerHTML = "";

            severity.textContent = "Invalid Image";
            severity.className = "severity high";

            confidenceMessage.innerHTML =
                "Please upload a clear image containing a single plant leaf.";

            warningBox.style.display = "block";

            warningBox.innerHTML = `
                <strong>Accepted Images:</strong>
                <ul>
                    <li>🌿 One plant leaf</li>
                    <li>📷 Good lighting</li>
                    <li>🎯 Leaf occupies most of the image</li>
                </ul>
            `;

            return;
        }

        // =====================================
        // Prediction Result
        // =====================================

        crop.textContent = data.crop;
        disease.textContent = data.disease;

        confidence.textContent = data.confidence + "%";

        confidenceBar.style.width =
            data.confidence + "%";

        confidenceBar.textContent =
            data.confidence + "%";

        description.textContent = data.description;

        cause.textContent = data.cause;

        createList(symptoms, data.symptoms);
        createList(treatment, data.treatment);
        createList(prevention, data.prevention);

        showTopPredictions(data.top_predictions);

        severity.textContent = data.severity;
        severity.className = "severity";

        if (data.severity === "Healthy") {

            severity.classList.add("healthy");

        } else if (data.severity === "Medium") {

            severity.classList.add("medium");

        } else {

            severity.classList.add("high");

        }

        if (data.confidence >= 90) {

            confidenceMessage.innerHTML =
                "✅ Excellent confidence. The prediction is highly reliable.";

            warningBox.style.display = "none";

        }

        else if (data.confidence >= 70) {

            confidenceMessage.innerHTML =
                "🟡 Good confidence. Verify symptoms before treatment.";

            warningBox.style.display = "none";

        }

        else {

            confidenceMessage.innerHTML =
                "⚠ Low confidence prediction.";

            warningBox.style.display = "block";

            warningBox.innerHTML = `
                <strong>Recommendation:</strong>
                <ul>
                    <li>Upload one leaf only.</li>
                    <li>Use good lighting.</li>
                    <li>Avoid blurry images.</li>
                    <li>Capture infected area clearly.</li>
                </ul>
            `;

        }

        resetBtn.classList.remove("hidden");

        result.classList.remove("hidden");

    }

    catch (error) {

        loading.classList.add("hidden");

        predictBtn.disabled = false;

        predictBtn.innerHTML = "🔍 Predict Disease";

        console.error(error);

        alert("Prediction failed. Please try again.");

    }

});
// ======================================================
// Reset Button
// ======================================================

resetBtn.addEventListener("click", () => {

    imageInput.value = "";

    preview.src = "";
    preview.style.display = "none";

    result.classList.add("hidden");

    resetBtn.classList.add("hidden");

    confidenceBar.style.width = "0%";
    confidenceBar.textContent = "0%";

    confidenceMessage.innerHTML = "";

    warningBox.style.display = "none";

    topPredictions.innerHTML = "";
    topPredictions.style.display = "none";

});


// ======================================================
// Camera
// ======================================================

cameraBtn.addEventListener("click", async () => {

    try {

        stream = await navigator.mediaDevices.getUserMedia({
            video: true
        });

        video.srcObject = stream;

        cameraModal.classList.remove("hidden");

    }

    catch (err) {

        console.error(err);

        alert("Unable to access camera.");

    }

});


// ======================================================
// Capture Image
// ======================================================

captureBtn.addEventListener("click", () => {

    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;

    const ctx = canvas.getContext("2d");

    ctx.drawImage(video, 0, 0, canvas.width, canvas.height);

    canvas.toBlob((blob) => {

        const file = new File(
            [blob],
            "captured_leaf.jpg",
            {
                type: "image/jpeg"
            }
        );

        const dt = new DataTransfer();

        dt.items.add(file);

        imageInput.files = dt.files;

        preview.src = URL.createObjectURL(file);

        preview.style.display = "block";

        if (stream) {

            stream.getTracks().forEach(track => track.stop());

        }

        cameraModal.classList.add("hidden");

    }, "image/jpeg");

});


// ======================================================
// Close Camera
// ======================================================

closeCameraBtn.addEventListener("click", () => {

    if (stream) {

        stream.getTracks().forEach(track => track.stop());

    }

    cameraModal.classList.add("hidden");

});
