// Interactive benchmark charts for docs/benchmark.md.
// Any element with class "benchmark-block" is auto-initialized. Re-runs on
// every Material "instant navigation" page change.

document$.subscribe(() => {
  document.querySelectorAll(".benchmark-block").forEach(initBenchmarkBlock);
});

function isDarkTheme() {
  return document.body.getAttribute("data-md-color-scheme") === "slate";
}

function initBenchmarkBlock(root) {
  const canvas = root.querySelector(".benchmark-canvas");
  const table = root.querySelector(".benchmark-table");
  const metricSelect = root.querySelector(".benchmark-metric");
  const seriesBoxes = [...root.querySelectorAll(".benchmark-series-input")];
  const viewToggle = root.querySelector(".benchmark-view-toggle");
  const meta = root.querySelector(".benchmark-meta");

  const xKey = root.dataset.x;
  const xLabel = root.dataset.xLabel || xKey;
  const baseline = root.dataset.baseline || null;
  const extraColumns = (root.dataset.extraColumns || "")
    .split(",")
    .filter(Boolean)
    .map((pair) => {
      const [key, label] = pair.split(":");
      return { key, label: label || key };
    });

  let chart = null;
  let data = null;
  let showTable = false;

  fetch(root.dataset.src)
    .then((r) => r.json())
    .then((json) => {
      data = json;
      meta.textContent = `backend: ${json.backend} · device: ${json.device} · generated ${json.generated}`;
      render();
    })
    .catch(() => {
      meta.textContent = "Could not load benchmark data.";
    });

  function metricValue(row, key) {
    const mode = metricSelect ? metricSelect.value : "time";
    if (mode === "speedup" && baseline) return row[baseline] / row[key];
    return row[key];
  }

  function metricLabel() {
    const mode = metricSelect ? metricSelect.value : "time";
    return mode === "speedup" ? `speedup vs ${baseline}` : "time [ms]";
  }

  function buildDatasets() {
    return seriesBoxes
      .filter((b) => b.checked)
      .map((b) => ({
        label: b.dataset.label,
        borderColor: b.dataset.color,
        backgroundColor: b.dataset.color,
        data: data.results.map((row) => ({ x: row[xKey], y: metricValue(row, b.value) })),
        tension: 0.25,
        pointRadius: 3,
      }));
  }

  function render() {
    if (!data) return;
    showTable ? renderTable() : renderChart();
  }

  function renderChart() {
    table.style.display = "none";
    canvas.style.display = "block";

    const dark = isDarkTheme();
    const gridColor = dark ? "rgba(255,255,255,0.08)" : "rgba(0,0,0,0.08)";
    const textColor = dark ? "#c9c9c9" : "#333";

    if (chart) chart.destroy();

    chart = new Chart(canvas.getContext("2d"), {
      type: "line",
      data: { datasets: buildDatasets() },
      options: {
        responsive: true,
        animation: false,
        interaction: { mode: "nearest", intersect: false },
        scales: {
          x: {
            type: "logarithmic",
            title: { display: true, text: xLabel, color: textColor },
            grid: { color: gridColor },
            ticks: { color: textColor },
          },
          y: {
            type: "logarithmic",
            title: { display: true, text: metricLabel(), color: textColor },
            grid: { color: gridColor },
            ticks: { color: textColor },
          },
        },
        plugins: { legend: { labels: { color: textColor } } },
      },
    });
  }

  function renderTable() {
    canvas.style.display = "none";
    table.style.display = "table";

    const activeBoxes = seriesBoxes.filter((b) => b.checked);
    const head =
      `<thead><tr><th>grid</th><th>${xLabel}</th>` +
      activeBoxes.map((b) => `<th>${b.dataset.label}</th>`).join("") +
      extraColumns.map((c) => `<th>${c.label}</th>`).join("") +
      `</tr></thead>`;

    const rows = data.results
      .map((row) => {
        const metricCells = activeBoxes
          .map((b) => {
            const v = metricValue(row, b.value);
            const suffix = (metricSelect ? metricSelect.value : "time") === "speedup" ? "×" : " ms";
            return `<td>${v.toFixed(v < 1 ? 4 : 2)}${suffix}</td>`;
          })
          .join("");
        const extraCells = extraColumns
          .map((c) => `<td>${row[c.key]}</td>`)
          .join("");
        const gridLabel = Array.isArray(row.n) ? row.n.join("×") : row[xKey];
        return `<tr><td>${gridLabel}</td><td>${row[xKey].toLocaleString()}</td>${metricCells}${extraCells}</tr>`;
      })
      .join("");

    table.innerHTML = head + `<tbody>${rows}</tbody>`;
  }

  if (metricSelect) metricSelect.addEventListener("change", render);
  seriesBoxes.forEach((b) => b.addEventListener("change", render));
  viewToggle.addEventListener("click", () => {
    showTable = !showTable;
    viewToggle.textContent = showTable ? "📈" : "▤";
    render();
  });

  new MutationObserver(() => {
    if (!showTable) render();
  }).observe(document.body, { attributes: true, attributeFilter: ["data-md-color-scheme"] });
}
