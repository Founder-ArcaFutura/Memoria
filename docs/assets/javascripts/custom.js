// Poll cluster rebuild status and display progress.
document.addEventListener('DOMContentLoaded', () => {
  const button = document.getElementById('rebuild-clusters');
  const statusEl = document.getElementById('cluster-status');
  if (!button || !statusEl) return;

  const pollStatus = () => {
    fetch('/clusters/status')
      .then((res) => res.json())
      .then((data) => {
        if (data.state === 'running') {
          statusEl.textContent = `Rebuilding ${data.current}/${data.total}`;
          setTimeout(pollStatus, 1000);
        } else if (data.state === 'error') {
          statusEl.textContent = `Error: ${data.error}`;
        } else if (data.state === 'complete') {
          statusEl.textContent = 'Rebuild complete';
        } else {
          statusEl.textContent = '';
        }
      })
      .catch((err) => {
        statusEl.textContent = `Error: ${err}`;
      });
  };

  button.addEventListener('click', () => {
    statusEl.textContent = 'Starting rebuild...';
    fetch('/clusters', { method: 'POST' })
      .catch((err) => {
        statusEl.textContent = `Error: ${err}`;
      });
    pollStatus();
  });
});
