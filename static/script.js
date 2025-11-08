document.getElementById('audioUpload').addEventListener('change', function() {
    const file = this.files[0];
    const instruction = document.getElementById('instruction');
    const fileNameElement = document.getElementById('fileName');
    const processButton = document.getElementById('processBtn');
    const audioPreview = document.getElementById('audioPreview');
    const audioPlayer = document.getElementById('audioPlayer');

    if (file) {
        // Display the selected file name
        fileNameElement.innerText = file.name;
        // Hide the instruction text
        instruction.style.display = 'none';
        // Enable the process button
        processButton.disabled = false;

        // Set up audio preview
        const fileURL = URL.createObjectURL(file);
        audioPlayer.src = fileURL;
        audioPreview.style.display = 'block';
    } else {
        // Reset if no file is selected
        fileNameElement.innerText = 'No file chosen';
        instruction.style.display = 'block';
        // Disable the process button
        processButton.disabled = true;
        // Hide audio preview
        audioPreview.style.display = 'none';
        audioPlayer.src = '';
    }
});

document.getElementById('uploadForm').addEventListener('submit', function(event) {
    event.preventDefault();

    const file = document.getElementById('audioUpload').files[0];
    const progressBarFill = document.getElementById('progressBar');
    const progressBarContainer = document.getElementById('progressBarContainer');
    const strokeType = document.getElementById('strokeType');

    if (file) {
        // Show the progress bar container
        progressBarContainer.style.display = 'block';

        // Reset the progress bar
        progressBarFill.style.width = '0%';

        // Create FormData to send file to the backend
        const formData = new FormData();
        formData.append('file', file);

        // Simulate the progress of processing the MP3 file with smoother animation
        let progress = 0;
        const totalSteps = 10;
        const stepDuration = 500; // ms per step for smoother animation
        const interval = setInterval(() => {
            progress += 10;
            progressBarFill.style.width = `${progress}%`;

            // Simulate completion at 100%
            if (progress >= 100) {
                clearInterval(interval);

                // Send the file to the backend
                fetch('/predict', {
                    method: 'POST',
                    body: formData,
                })
                .then(response => response.json())
                .then(data => {
                    if (data.error) {
                        strokeType.innerText = "Error: " + data.error;
                    } else {
                        strokeType.innerText = data.predicted_stroke;  // Display predicted stroke
                    }
                })
                .catch(error => {
                    console.error('Error:', error);
                    if (error.name === 'TypeError' && error.message.includes('fetch')) {
                        strokeType.innerText = 'Network error: Please check your connection and try again.';
                    } else {
                        strokeType.innerText = 'Error processing file: ' + error.message;
                    }
                })
                .finally(() => {
                    // Hide the progress bar after processing is complete
                    progressBarContainer.style.display = 'none';
                });
            }
        }, stepDuration); // Smoother progress animation
    } else {
        alert('Please upload an audio file.');
    }
});
