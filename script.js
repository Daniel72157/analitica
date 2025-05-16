document.getElementById("predictForm").addEventListener("submit", function (e) {
    e.preventDefault();
  
    const formData = new FormData(e.target);
  
    // Obtener el modelo seleccionado
    const selectedModel = document.getElementById("modelSelect").value;
  
    fetch("/predict", {
      method: "POST",
      body: new URLSearchParams(formData) + "&modelSelect=" + selectedModel // Añadir el modelo al body de la petición
    })
      .then(response => response.json())
      .then(data => {
        document.getElementById("results").style.display = "block";
  
        // Mostrar el resultado para el modelo seleccionado
        let result = data[selectedModel] === 1 ? "FGR" : "Normal";
        document.getElementById("results").innerHTML = `
          <h2>Predicción usando el modelo: ${selectedModel}</h2>
          <p><strong>Resultado:</strong> ${result}</p>
        `;
      })
      .catch(error => {
        alert("Error en la predicción. Revisa la consola.");
        console.error(error);
      });
  });
  
document.getElementById("batchForm").addEventListener("submit", function (e) {
  e.preventDefault();

  const fileInput = document.getElementById("csvFile");
  const modelSelect = document.getElementById("modelSelect"); // ya existente
  const selectedModel = modelSelect.value;

  if (!fileInput.files.length) {
    alert("Por favor selecciona un archivo CSV.");
    return;
  }

  const formData = new FormData();
  formData.append("csv", fileInput.files[0]);
  formData.append("model", selectedModel);

  fetch("/batch_predict", {
    method: "POST",
    body: formData
  })
    .then(res => res.json())
    .then(data => {
      // Mostrar los resultados en una tabla HTML
      const batchResultContainer = document.getElementById("batchResult");
      batchResultContainer.innerHTML = `
        <strong>Exactitud:</strong> ${data.accuracy}%<br>
        <strong>Matriz de Confusión:</strong>
        <table border="1">
          <thead>
            <tr>
              <th></th>
              <th>Predicción Positiva</th>
              <th>Predicción Negativa</th>
            </tr>
          </thead>
          <tbody>
            <tr>
              <td>Real Positivo</td>
              <td>${data.confusion_matrix[0][0]}</td>
              <td>${data.confusion_matrix[0][1]}</td>
            </tr>
            <tr>
              <td>Real Negativo</td>
              <td>${data.confusion_matrix[1][0]}</td>
              <td>${data.confusion_matrix[1][1]}</td>
            </tr>
          </tbody>
        </table>
      `;
    })
    .catch(err => {
      console.error(err);
      document.getElementById("batchResult").innerHTML = `<p style="color:red;">Error al procesar el archivo.</p>`;
    });
});

