import React, {useState, useEffect, useMemo} from 'react';
import {Line} from 'react-chartjs-2';
import 'chart.js/auto';
import {useColorMode} from '@docusaurus/theme-common';
import useBaseUrl from '@docusaurus/useBaseUrl';
import styles from './styles.module.css';

/**
 * Interactive benchmark chart. Fetches a JSON results file (produced by the
 * benchmark scripts under benchmark/) and renders a toggleable line chart /
 * table, matching the current color scheme.
 *
 * @param {string} src - path to the JSON data file, relative to the site's static dir
 * @param {string} xKey - field name in each result row used for the x-axis
 * @param {string} xLabel - axis label for xKey
 * @param {string} [baseline] - field name to compute "speedup" against (enables the metric selector)
 * @param {{key: string, label: string, color: string}[]} series - candidate series, each togglable
 * @param {{key: string, label: string}[]} [extraColumns] - extra raw columns shown only in table view
 */
export default function BenchmarkChart({src, xKey, xLabel, baseline, series, extraColumns = []}) {
  const {colorMode} = useColorMode();
  const dataUrl = useBaseUrl(src);

  const [data, setData] = useState(null);
  const [error, setError] = useState(false);
  const [activeSeries, setActiveSeries] = useState(() => new Set(series.map((s) => s.key)));
  const [metric, setMetric] = useState('time');
  const [showTable, setShowTable] = useState(false);

  useEffect(() => {
    fetch(dataUrl)
      .then((r) => r.json())
      .then(setData)
      .catch(() => setError(true));
  }, [dataUrl]);

  function toggleSeries(key) {
    setActiveSeries((prev) => {
      const next = new Set(prev);
      if (next.has(key)) next.delete(key);
      else next.add(key);
      return next;
    });
  }

  function metricValue(row, key) {
    if (metric === 'speedup' && baseline) return row[baseline] / row[key];
    return row[key];
  }

  const activeSeriesList = series.filter((s) => activeSeries.has(s.key));

  const chartData = useMemo(() => {
    if (!data) return null;
    return {
      datasets: activeSeriesList.map((s) => ({
        label: s.label,
        borderColor: s.color,
        backgroundColor: s.color,
        data: data.results.map((row) => ({x: row[xKey], y: metricValue(row, s.key)})),
        tension: 0.25,
        pointRadius: 3,
      })),
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [data, activeSeries, metric]);

  if (error) {
    return <div className={styles.meta}>Could not load benchmark data.</div>;
  }
  if (!data) {
    return <div className={styles.meta}>Loading benchmark…</div>;
  }

  const dark = colorMode === 'dark';
  const gridColor = dark ? 'rgba(255,255,255,0.08)' : 'rgba(0,0,0,0.08)';
  const textColor = dark ? '#c9c9c9' : '#333';

  const options = {
    responsive: true,
    animation: false,
    interaction: {mode: 'nearest', intersect: false},
    scales: {
      x: {
        type: 'logarithmic',
        title: {display: true, text: xLabel, color: textColor},
        grid: {color: gridColor},
        ticks: {color: textColor},
      },
      y: {
        type: 'logarithmic',
        title: {
          display: true,
          text: metric === 'speedup' ? `speedup vs ${baseline}` : 'time [ms]',
          color: textColor,
        },
        grid: {color: gridColor},
        ticks: {color: textColor},
      },
    },
    plugins: {legend: {labels: {color: textColor}}},
  };

  return (
    <div className={styles.block}>
      <div className={styles.controls}>
        {baseline && (
          <label>
            Metric{' '}
            <select value={metric} onChange={(e) => setMetric(e.target.value)}>
              <option value="time">Time [ms]</option>
              <option value="speedup">Speedup vs {baseline}</option>
            </select>
          </label>
        )}

        {series.map((s) => (
          <label key={s.key} className={styles.series}>
            <input
              type="checkbox"
              checked={activeSeries.has(s.key)}
              onChange={() => toggleSeries(s.key)}
            />{' '}
            {s.label}
          </label>
        ))}

        <button
          className={styles.viewToggle}
          onClick={() => setShowTable((v) => !v)}
          title="Toggle chart/table view"
        >
          {showTable ? '📈' : '▤'}
        </button>
      </div>

      {showTable ? (
        <table className={styles.table}>
          <thead>
            <tr>
              <th>grid</th>
              <th>{xLabel}</th>
              {activeSeriesList.map((s) => (
                <th key={s.key}>{s.label}</th>
              ))}
              {extraColumns.map((c) => (
                <th key={c.key}>{c.label}</th>
              ))}
            </tr>
          </thead>
          <tbody>
            {data.results.map((row, i) => (
              <tr key={i}>
                <td>{Array.isArray(row.n) ? row.n.join('×') : row[xKey]}</td>
                <td>{row[xKey].toLocaleString()}</td>
                {activeSeriesList.map((s) => {
                  const v = metricValue(row, s.key);
                  const suffix = metric === 'speedup' ? '×' : ' ms';
                  return (
                    <td key={s.key}>
                      {v.toFixed(v < 1 ? 4 : 2)}
                      {suffix}
                    </td>
                  );
                })}
                {extraColumns.map((c) => (
                  <td key={c.key}>{row[c.key]}</td>
                ))}
              </tr>
            ))}
          </tbody>
        </table>
      ) : (
        <Line data={chartData} options={options} />
      )}

      <div className={styles.meta}>
        backend: {data.backend} · device: {data.device} · generated {data.generated}
      </div>
    </div>
  );
}
