
function startDetector(){
    const status = document.getElementById('status');
    if(status) status.innerText = 'Status: Please run `python main.py` in your project root.';
    alert('To start detector, run: python main.py');
}
