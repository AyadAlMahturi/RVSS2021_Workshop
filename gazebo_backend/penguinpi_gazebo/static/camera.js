document.addEventListener("DOMContentLoaded", () => {
  const result = document.querySelector('.camera')
  const reader = new FileReader();

  function updateCamera() {
    fetch('/camera/get', { cache: "no-store" })
      .then(response => response.blob())
      .then(blob => {
        reader.readAsDataURL(blob);
        setTimeout(updateCamera, 30);
      });
  }

  reader.onloadend = function() {
     result.src = reader.result;
  }

  if (result) {
    setTimeout(updateCamera, 30);
  };
})
