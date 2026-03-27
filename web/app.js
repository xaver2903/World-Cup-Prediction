function parseCSV(text) {
  const lines = text.trim().split(/\r?\n/);
  const headers = lines[0].split(',');
  return lines.slice(1).map((line) => {
    const values = line.split(',');
    return headers.reduce((row, header, index) => {
      row[header] = values[index];
      return row;
    }, {});
  });
}

function toPct(value) {
  const number = Number(value);
  return `${(number * 100).toFixed(1)}%`;
}

function renderWinner(rows) {
  const winner = rows[0];
  if (!winner) return;

  document.getElementById('winner-team').textContent = winner.team;
  document.getElementById('winner-probability').textContent = toPct(
    winner.win_world_cup,
  );

  document.getElementById('winner-loading').classList.add('hidden');
  document.getElementById('winner-content').classList.remove('hidden');
}

function renderChampionTable(rows) {
  const tbody = document.querySelector('#champion-table tbody');
  rows.slice(0, 15).forEach((row, index) => {
    const tr = document.createElement('tr');
    tr.innerHTML = `
      <td>${index + 1}</td>
      <td>${row.team}</td>
      <td>${toPct(row.win_world_cup)}</td>
    `;
    tbody.appendChild(tr);
  });
}

function renderStageTable(rows) {
  const tbody = document.querySelector('#stage-table tbody');
  rows.slice(0, 20).forEach((row) => {
    const tr = document.createElement('tr');
    tr.innerHTML = `
      <td>${row.team}</td>
      <td>${toPct(row.advance_group)}</td>
      <td>${toPct(row.reach_r16)}</td>
      <td>${toPct(row.reach_qf)}</td>
      <td>${toPct(row.reach_sf)}</td>
      <td>${toPct(row.reach_final)}</td>
      <td>${toPct(row.win_world_cup)}</td>
    `;
    tbody.appendChild(tr);
  });
}

function renderGroupTable(rows) {
  const tbody = document.querySelector('#group-table tbody');
  rows.forEach((row) => {
    const tr = document.createElement('tr');
    tr.innerHTML = `
      <td>${row.group}</td>
      <td>${row.home_team}</td>
      <td>${row.away_team}</td>
      <td>${toPct(row.proba_home_win)}</td>
      <td>${toPct(row.proba_draw)}</td>
      <td>${toPct(row.proba_away_win)}</td>
    `;
    tbody.appendChild(tr);
  });
}

async function loadCSV(path) {
  const response = await fetch(path);
  if (!response.ok) {
    throw new Error(`Failed to load ${path}`);
  }
  return parseCSV(await response.text());
}

async function init() {
  try {
    const [winnerRows, stageRows, groupRows] = await Promise.all([
      loadCSV('../outputs/most_likely_winner.csv'),
      loadCSV('../outputs/team_stage_probabilities.csv'),
      loadCSV('../data/processed/world_cup_group_predictions.csv'),
    ]);

    renderWinner(winnerRows);
    renderChampionTable(stageRows);
    renderStageTable(stageRows);
    renderGroupTable(groupRows);
  } catch (error) {
    document.getElementById('winner-loading').textContent =
      `Unable to load dashboard data: ${error.message}`;
  }
}

init();
