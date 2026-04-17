  document.addEventListener('DOMContentLoaded', function() {
    
    const form = document.getElementById('diabetes-form');
    
    form.addEventListener('submit', async function(event) {
      event.preventDefault();
      event.stopPropagation();
      
      
      const submitBtn = document.getElementById('submitBtn');
      
      // Validate all required fields
      const requiredFields = form.querySelectorAll('[required]');
      let isValid = true;
      
      for (let field of requiredFields) {
        if (!field.value) {
          alert(`Please fill out: ${field.name || field.id}`);
          isValid = false;
          return;
        }
      }
      
      if (!isValid) return;
      
      
      const formData = new FormData(form);
      const jsonData = {};
      
      for (let [key, value] of formData.entries()) {
        if (key === 'BMI') {
          jsonData[key] = parseFloat(value);
        } else {
          jsonData[key] = parseInt(value);
        }
      }
      
      console.log('Sending data to API:', jsonData);
      
     
      const resultPanel = document.getElementById('result-panel');
      const loadingDiv = document.getElementById('loading');
      const resultsContent = document.getElementById('results-content');
      
      resultPanel.style.display = 'block';
      loadingDiv.style.display = 'block';
      resultsContent.style.display = 'none';
      
      submitBtn.disabled = true;
      submitBtn.textContent = 'Predicting...';
      
      try {
        
        const response = await fetch('/predict', {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json',
          },
          body: JSON.stringify(jsonData)
        });
        
        console.log('Response status:', response.status);
        
        const result = await response.json();
        console.log('Response data:', result);
        
        if (response.ok) {
          const probability = result.probability * 100;
          const isHighRisk = result.prediction === 1;
          
          const riskBar = document.getElementById('risk-bar');
          if (riskBar) {
            riskBar.style.width = `${probability}%`;
            riskBar.textContent = `${probability.toFixed(1)}%`;
          }
          
          document.getElementById('meta-pred').textContent = isHighRisk ? 'High Risk' : 'Low Risk';
          document.getElementById('meta-conf').textContent = `${probability.toFixed(1)}%`;
          
          if (isHighRisk) {
            document.getElementById('result-icon').textContent = '⚠️';
            document.getElementById('result-title').textContent = ' Elevated Diabetes Risk Detected';
            document.getElementById('result-subtitle').textContent = 'Consult a healthcare provider for preventive care';
            document.getElementById('meta-rec').textContent = 'Schedule a checkup with your doctor immediately';
            document.getElementById('meta-pred').style.color = '#dc3545';
          } else {
            document.getElementById('result-icon').textContent = '✓';
            document.getElementById('result-title').textContent = ' Low Diabetes Risk';
            document.getElementById('result-subtitle').textContent = 'Maintain healthy lifestyle habits';
            document.getElementById('meta-rec').textContent = 'Continue healthy habits and regular checkups';
            document.getElementById('meta-pred').style.color = '#28a745';
          }
          
          resultsContent.style.display = 'block';
          
        
          resultPanel.scrollIntoView({ behavior: 'smooth' });
        } else {
          throw new Error(result.detail || 'Prediction failed');
        }
      } catch (error) {
        console.error('Error:', error);
        alert('Error making prediction: ' + error.message);
        resultPanel.style.display = 'none';
      } finally {
        loadingDiv.style.display = 'none';
        submitBtn.disabled = false;
        submitBtn.textContent = 'Predict Diabetes Risk';
      }
    });
  });
  
  function resetForm() {
    const form = document.getElementById('diabetes-form');
    form.reset();
    const resultPanel = document.getElementById('result-panel');
    resultPanel.style.display = 'none';
    const resultsContent = document.getElementById('results-content');
    resultsContent.style.display = 'none';
    console.log('Form reset');
  }