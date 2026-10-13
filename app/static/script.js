// FantasyScore AI - Real-Time Engine & Live Simulator Client Logic

let allPlayersCache = [];
let baselinePlayerStats = {};
let realtimeDebounceTimer = null;
let simulationInterval = null;
let simulationState = {
  active: false,
  scenario: "chase_top_order",
  stepIndex: 0,
  maxSteps: 18,
  currentRuns: 0,
  currentBalls: 0,
  fours: 0,
  sixes: 0,
  wickets: 0,
  overs: 0,
  runsConceded: 0,
  scoreHistory: []
};

document.addEventListener("DOMContentLoaded", () => {
  initModeTabs();
  initLivePlayerSearch();
  initPlayerAutoFill();
  initQuickChips();
  initRealtimeControls();
  initWhatIfScenarios();
  initLiveMatchSimulator();
  initLeaderboardSearch();
  initTableSorting();
  initSquadOptimizerFilters();

  // Initial load auto-trigger to ensure real-time analytics are visible
  const form = document.getElementById("predictionForm");
  if (form) {
    runRealtimePrediction();
  }
});

// ==========================================================================
// 1. Mode Navigation Tabs
// ==========================================================================
function initModeTabs() {
  const tabs = document.querySelectorAll(".mode-tab");
  tabs.forEach(tab => {
    tab.addEventListener("click", () => {
      tabs.forEach(t => t.classList.remove("active"));
      tab.classList.add("active");

      const targetId = tab.getAttribute("data-target");
      document.querySelectorAll(".tab-view-content").forEach(view => {
        view.classList.remove("active");
      });

      const targetView = document.getElementById(targetId);
      if (targetView) {
        targetView.classList.add("active");
      }

      // If switching to simulator, render initial trajectory chart
      if (targetId === "simulator-mode-view") {
        renderTrajectoryChart();
      }
    });
  });
}

// ==========================================================================
// 2. Real-Time Synchronized Dual Controls (Sliders <-> Number Inputs)
// ==========================================================================
function initRealtimeControls() {
  const form = document.getElementById("predictionForm");
  if (!form) return;

  const sliderPairs = [
    { numId: "input_batting_runs", rangeId: "range_batting_runs", badgeId: "val_batting_runs" },
    { numId: "input_batting_strike_rate", rangeId: "range_batting_strike_rate", badgeId: "val_batting_strike_rate" },
    { numId: "input_batting_average", rangeId: "range_batting_average", badgeId: "val_batting_average" },
    { numId: "input_boundaries", rangeId: "range_boundaries", badgeId: "val_boundaries" },
    { numId: "input_bowling_wickets", rangeId: "range_bowling_wickets", badgeId: "val_bowling_wickets" },
    { numId: "input_bowling_economy", rangeId: "range_bowling_economy", badgeId: "val_bowling_economy" },
    { numId: "input_bowling_average", rangeId: "range_bowling_average", badgeId: "val_bowling_average" },
    { numId: "input_overs_bowled", rangeId: "range_overs_bowled", badgeId: "val_overs_bowled" },
    { numId: "input_recent_fantasy_points", rangeId: "range_recent_fantasy_points", badgeId: "val_recent_fantasy_points" },
    { numId: "input_recent_average_points", rangeId: "range_recent_average_points", badgeId: "val_recent_average_points" },
    { numId: "input_player_form", rangeId: "range_player_form", badgeId: "val_player_form" },
    { numId: "input_playing_time_percentage", rangeId: "range_playing_time_percentage", badgeId: "val_playing_time_percentage" },
    { numId: "input_opponent_strength", rangeId: "range_opponent_strength", badgeId: "val_opponent_strength" }
  ];

  sliderPairs.forEach(pair => {
    const numInput = document.getElementById(pair.numId);
    const rangeInput = document.getElementById(pair.rangeId);
    const badge = document.getElementById(pair.badgeId);

    if (numInput && rangeInput) {
      // Sync Range -> Number
      rangeInput.addEventListener("input", (e) => {
        const val = e.target.value;
        numInput.value = val;
        if (badge) badge.textContent = val;
        triggerDebouncedPrediction();
      });

      // Sync Number -> Range
      numInput.addEventListener("input", (e) => {
        const val = e.target.value;
        rangeInput.value = val;
        if (badge) badge.textContent = val;
        triggerDebouncedPrediction();
      });
    }
  });

  // Watch dropdown selects (Team, Opponent, Role, Home/Away)
  const selects = form.querySelectorAll("select, input[type='text']");
  selects.forEach(elem => {
    elem.addEventListener("change", () => triggerDebouncedPrediction());
  });

  // Manual Trigger Button
  const predictBtn = document.getElementById("predictBtn");
  if (predictBtn) {
    predictBtn.addEventListener("click", () => runRealtimePrediction());
  }

  // Reset Button
  const resetBtn = document.getElementById("resetPlayerStatsBtn");
  if (resetBtn) {
    resetBtn.addEventListener("click", () => {
      if (Object.keys(baselinePlayerStats).length > 0) {
        populateFormFields(baselinePlayerStats);
        runRealtimePrediction();
      }
    });
  }
}

function triggerDebouncedPrediction() {
  clearTimeout(realtimeDebounceTimer);
  realtimeDebounceTimer = setTimeout(() => {
    runRealtimePrediction();
  }, 40); // 40ms ultra-fast responsive debounce
}

// ==========================================================================
// 3. Live Player Search & Autocomplete
// ==========================================================================
async function initLivePlayerSearch() {
  const searchInput = document.getElementById("playerSearchInput");
  const resultsBox = document.getElementById("playerSearchResults");
  const clearBtn = document.getElementById("clearPlayerSearchBtn");

  if (!searchInput || !resultsBox) return;

  try {
    const resp = await fetch("/api/players");
    const data = await resp.json();
    if (data.success && Array.isArray(data.players)) {
      allPlayersCache = data.players;
    }
  } catch (e) {
    console.warn("Could not pre-load players cache", e);
  }

  let activeIndex = -1;

  searchInput.addEventListener("input", () => {
    const query = searchInput.value.trim().toLowerCase();
    if (clearBtn) clearBtn.style.display = query.length > 0 ? "flex" : "none";

    if (!query) {
      resultsBox.style.display = "none";
      resultsBox.innerHTML = "";
      activeIndex = -1;
      return;
    }

    const matches = allPlayersCache.filter(p => 
      p.name.toLowerCase().includes(query) || 
      p.team.toLowerCase().includes(query) ||
      p.role.toLowerCase().includes(query)
    ).slice(0, 10);

    if (matches.length === 0) {
      resultsBox.innerHTML = `<div style="padding: 12px; text-align: center; color: var(--text-muted); font-size: 13px;">No players found matching "${searchInput.value}"</div>`;
      resultsBox.style.display = "block";
      activeIndex = -1;
      return;
    }

    resultsBox.innerHTML = matches.map((p, idx) => `
      <div class="search-result-item" data-player="${p.name}" data-idx="${idx}">
        <div class="search-item-left">
          <div class="player-avatar-wrapper search-avatar-wrapper">
            <img src="${p.image_url || 'https://ui-avatars.com/api/?name=' + encodeURIComponent(p.name) + '&background=0f172a&color=06b6d4&size=128&bold=true'}" 
                 alt="${p.name}" 
                 class="player-avatar-img search-avatar-img" 
                 loading="lazy" 
                 onerror="this.style.display='none'; this.nextElementSibling.style.display='flex';">
            <div class="search-item-avatar player-avatar-fallback" style="display: none;">
              ${p.initials || p.name.substring(0, 2).toUpperCase()}
            </div>
          </div>
          <div>
            <div class="search-item-name">${highlightMatch(p.name, query)} <span class="player-id-tag">#${p.player_id || ''}</span></div>
            <div class="search-item-meta">${p.team} &bull; ${p.role}</div>
          </div>
        </div>
        <span class="search-item-badge">Avg: ${Number(p.avg_points).toFixed(1)} pts</span>
      </div>
    `).join("");

    resultsBox.style.display = "block";
    activeIndex = -1;

    resultsBox.querySelectorAll(".search-result-item").forEach(item => {
      item.addEventListener("click", () => {
        const playerName = item.getAttribute("data-player");
        selectPlayerByName(playerName);
        searchInput.value = playerName;
        resultsBox.style.display = "none";
        if (clearBtn) clearBtn.style.display = "flex";
      });
    });
  });

  searchInput.addEventListener("keydown", (e) => {
    const items = resultsBox.querySelectorAll(".search-result-item");

    if (e.key === "ArrowDown") {
      if (items.length === 0 || resultsBox.style.display === "none") return;
      e.preventDefault();
      activeIndex = (activeIndex + 1) % items.length;
      updateActiveItem(items, activeIndex);
    } else if (e.key === "ArrowUp") {
      if (items.length === 0 || resultsBox.style.display === "none") return;
      e.preventDefault();
      activeIndex = (activeIndex - 1 + items.length) % items.length;
      updateActiveItem(items, activeIndex);
    } else if (e.key === "Enter") {
      e.preventDefault();
      if (resultsBox.style.display !== "none" && items.length > 0) {
        const targetItem = activeIndex >= 0 && activeIndex < items.length ? items[activeIndex] : items[0];
        targetItem.click();
      } else {
        const query = searchInput.value.trim().toLowerCase();
        const match = allPlayersCache.find(p => p.name.toLowerCase() === query) ||
                      allPlayersCache.find(p => p.name.toLowerCase().includes(query));
        if (match) {
          selectPlayerByName(match.name);
          searchInput.value = match.name;
          resultsBox.style.display = "none";
          if (clearBtn) clearBtn.style.display = "flex";
        }
      }
    } else if (e.key === "Escape") {
      resultsBox.style.display = "none";
    }
  });

  if (clearBtn) {
    clearBtn.addEventListener("click", () => {
      searchInput.value = "";
      resultsBox.style.display = "none";
      clearBtn.style.display = "none";
      searchInput.focus();
    });
  }

  document.addEventListener("click", (e) => {
    if (!searchInput.contains(e.target) && !resultsBox.contains(e.target)) {
      resultsBox.style.display = "none";
    }
  });
}

function updateActiveItem(items, index) {
  items.forEach((item, idx) => {
    if (idx === index) {
      item.classList.add("selected");
      item.scrollIntoView({ block: "nearest" });
    } else {
      item.classList.remove("selected");
    }
  });
}

function highlightMatch(text, query) {
  if (!query) return text;
  const index = text.toLowerCase().indexOf(query);
  if (index === -1) return text;
  return text.substring(0, index) + 
         `<span style="color: var(--accent-cyan); font-weight: 700; text-decoration: underline;">${text.substring(index, index + query.length)}</span>` + 
         text.substring(index + query.length);
}

// ==========================================================================
// 4. Quick Star Chips & Dropdown Selection
// ==========================================================================
function initQuickChips() {
  const chips = document.querySelectorAll(".quick-chip");
  chips.forEach(chip => {
    chip.addEventListener("click", () => {
      const playerName = chip.getAttribute("data-player");
      const searchInput = document.getElementById("playerSearchInput");
      const clearBtn = document.getElementById("clearPlayerSearchBtn");
      
      if (searchInput) searchInput.value = playerName;
      if (clearBtn) clearBtn.style.display = "flex";
      
      selectPlayerByName(playerName);
    });
  });
}

function initPlayerAutoFill() {
  const playerSelect = document.getElementById("playerSelect");
  if (!playerSelect) return;

  playerSelect.addEventListener("change", (e) => {
    const playerName = e.target.value;
    if (playerName) {
      const searchInput = document.getElementById("playerSearchInput");
      const clearBtn = document.getElementById("clearPlayerSearchBtn");
      if (searchInput) searchInput.value = playerName;
      if (clearBtn) clearBtn.style.display = "flex";
      
      selectPlayerByName(playerName);
    }
  });
}

function updatePlayerProfileUI(playerData) {
  if (!playerData) return;
  const name = playerData.player_name || playerData.name || "";
  const id = playerData.player_id || "";
  const img = playerData.image_url || "";
  const words = name.split(" ");
  const initials = playerData.initials || (words.length > 0 ? words.map(w => w[0]).join("").substring(0, 2).toUpperCase() : "PL");
  const team = playerData.team || "";
  const role = playerData.player_role || playerData.role || "";

  // 1. Update Simulator Header
  const simPlayerName = document.getElementById("simPlayerName");
  const simPlayerId = document.getElementById("simPlayerId");
  const simPlayerImage = document.getElementById("simPlayerImage");
  const simPlayerAvatar = document.getElementById("simPlayerAvatar");

  if (simPlayerName && name) simPlayerName.textContent = name;
  if (simPlayerId && id) simPlayerId.textContent = `#${id}`;
  if (simPlayerImage && img) {
    simPlayerImage.src = img;
    simPlayerImage.style.display = "block";
  }
  if (simPlayerAvatar) simPlayerAvatar.textContent = initials;

  // 2. Update Right Column Prediction Hub Header Card
  const hubPlayerName = document.getElementById("hubPlayerName");
  const hubPlayerId = document.getElementById("hubPlayerId");
  const hubPlayerTeam = document.getElementById("hubPlayerTeam");
  const hubPlayerRole = document.getElementById("hubPlayerRole");
  const hubPlayerImage = document.getElementById("hubPlayerImage");
  const hubPlayerFallback = document.getElementById("hubPlayerFallback");

  if (hubPlayerName && name) hubPlayerName.textContent = name;
  if (hubPlayerId && id) hubPlayerId.textContent = `#${id}`;
  if (hubPlayerTeam && team) hubPlayerTeam.textContent = team;
  if (hubPlayerRole && role) hubPlayerRole.textContent = role;
  if (hubPlayerImage && img) {
    hubPlayerImage.src = img;
    hubPlayerImage.style.display = "block";
  }
  if (hubPlayerFallback) hubPlayerFallback.textContent = initials;
}

async function selectPlayerByName(playerName) {
  if (!playerName) return;

  const playerSelect = document.getElementById("playerSelect");
  if (playerSelect) {
    playerSelect.value = playerName;
  }

  try {
    const resp = await fetch(`/api/player/${encodeURIComponent(playerName)}`);
    if (!resp.ok) throw new Error("Failed to load player statistics");
    
    const data = await resp.json();
    if (data.success && data.stats) {
      baselinePlayerStats = { ...data.stats };
      updatePlayerProfileUI(data.stats);
      populateFormFields(data.stats);
      flashAutoFillBanner(playerName);
      await runRealtimePrediction();
    }
  } catch (err) {
    console.warn("Auto-fill error:", err);
  }
}

function populateFormFields(stats) {
  for (const [key, value] of Object.entries(stats)) {
    const input = document.getElementById(`input_${key}`) || document.querySelector(`[name="${key}"]`);
    const range = document.getElementById(`range_${key}`);
    const badge = document.getElementById(`val_${key}`);

    if (input) {
      if (input.tagName === "SELECT") {
        let optionExists = false;
        for (let opt of input.options) {
          if (opt.value.toLowerCase() === String(value).toLowerCase()) {
            input.value = opt.value;
            optionExists = true;
            break;
          }
        }
        if (!optionExists && value) {
          const newOpt = new Option(value, value, true, true);
          input.add(newOpt);
        }
      } else {
        const valFormatted = typeof value === 'number' ? (Number.isInteger(value) ? value : value.toFixed(2)) : value;
        input.value = valFormatted;
        if (range) range.value = valFormatted;
        if (badge) badge.textContent = valFormatted;
      }
      
      input.classList.add("field-highlight");
      setTimeout(() => input.classList.remove("field-highlight"), 900);
    }
  }
}

function flashAutoFillBanner(name) {
  const banner = document.getElementById("autoFillBanner");
  if (banner) {
    banner.style.display = "block";
    banner.innerHTML = `✓ Real-time baseline loaded for <strong>${name}</strong>`;
    setTimeout(() => {
      banner.style.display = "none";
    }, 3500);
  }
}

// ==========================================================================
// 5. Asynchronous Real-Time Score Prediction
// ==========================================================================
async function runRealtimePrediction() {
  const form = document.getElementById("predictionForm");
  if (!form) return;

  const formData = new FormData(form);
  const payload = {};
  formData.forEach((value, key) => {
    payload[key] = value;
  });

  const t0 = performance.now();

  try {
    const resp = await fetch("/api/predict", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload)
    });

    const latency = (performance.now() - t0).toFixed(1);
    const result = await resp.json();

    if (result.success) {
      displayRealtimeResult(result.prediction, result.breakdown, result.analytics, latency);
    }
  } catch (err) {
    console.error("Realtime Prediction failed:", err);
  }
}

function displayRealtimeResult(score, breakdown, analytics, clientLatency) {
  const scoreDisplay = document.getElementById("predictedScoreDisplay");
  const tierDisplay = document.getElementById("tierIndicator");
  const factorsContainer = document.getElementById("explainFactorsContainer");
  const confidenceDisplay = document.getElementById("confidenceRangeDisplay");
  const probDisplay = document.getElementById("selectionProbDisplay");
  const probBar = document.getElementById("selectionProbBar");
  const latencyDisplay = document.getElementById("predLatencyDisplay");
  const headerLatency = document.getElementById("headerLatencyChip");
  const navLatency = document.getElementById("navbarLatencyPill");

  if (!scoreDisplay) return;

  // Latency indicators
  const serverLatency = analytics && analytics.latency_ms ? analytics.latency_ms : clientLatency;
  if (latencyDisplay) latencyDisplay.textContent = `Inference: ${serverLatency}ms`;
  if (headerLatency) headerLatency.textContent = `⚡ Latency: ${serverLatency}ms`;
  if (navLatency) navLatency.textContent = `~${serverLatency}ms`;

  // Animate score count
  const prevScore = parseFloat(scoreDisplay.textContent) || 0;
  animateNumber(scoreDisplay, prevScore, Number(score));

  scoreDisplay.classList.add("score-updated");
  setTimeout(() => scoreDisplay.classList.remove("score-updated"), 400);

  // Confidence Range
  if (confidenceDisplay && analytics) {
    confidenceDisplay.innerHTML = `Confidence Interval: <strong>${Number(analytics.confidence_low).toFixed(1)} - ${Number(analytics.confidence_high).toFixed(1)} pts</strong>`;
  }

  // Selection Probability
  if (probDisplay && probBar && analytics) {
    probDisplay.textContent = `${analytics.selection_prob}%`;
    probBar.style.width = `${analytics.selection_prob}%`;
  }

  // Tier Badge
  if (tierDisplay && analytics) {
    tierDisplay.className = `tier-indicator ${analytics.tier_class || ''}`;
    tierDisplay.textContent = analytics.tier_title || '★ High Impact Player';
  }

  // Explainability Bars
  if (factorsContainer && breakdown && breakdown.length > 0) {
    factorsContainer.innerHTML = breakdown.map(item => `
      <div class="factor-item">
        <div class="factor-meta">
          <span class="f-name">${item.name}</span>
          <span class="f-val">${item.value} (${item.importance}% impact)</span>
        </div>
        <div class="factor-bar-bg">
          <div class="factor-bar-fill" style="width: ${Math.min(100, Math.max(8, item.importance))}%;"></div>
        </div>
      </div>
    `).join("");
  }

  // Update Player Profile Card & Avatar
  if (analytics && analytics.player_name) {
    updatePlayerProfileUI(analytics);
  }
}

function animateNumber(element, startVal, endVal, duration = 300) {
  const startTime = performance.now();
  
  function update(currentTime) {
    const elapsed = currentTime - startTime;
    const progress = Math.min(elapsed / duration, 1);
    const easeOut = 1 - Math.pow(1 - progress, 3);
    const current = startVal + (endVal - startVal) * easeOut;
    
    element.textContent = current.toFixed(2);
    
    if (progress < 1) {
      requestAnimationFrame(update);
    } else {
      element.textContent = Number(endVal).toFixed(2);
    }
  }
  
  requestAnimationFrame(update);
}

// ==========================================================================
// 6. 1-Click "What-If" Match Scenarios
// ==========================================================================
function initWhatIfScenarios() {
  const scenarioCards = document.querySelectorAll(".scenario-card");
  const scenarios = {
    half_century: {
      batting_runs: 54,
      batting_strike_rate: 180.0,
      boundaries: 8,
      player_form: 88,
      playing_time_percentage: 100,
      bowling_wickets: 0,
      overs_bowled: 0
    },
    captain_knock: {
      batting_runs: 82,
      batting_strike_rate: 164.0,
      boundaries: 10,
      player_form: 95,
      playing_time_percentage: 100,
      recent_fantasy_points: 92,
      bowling_wickets: 0
    },
    death_finisher: {
      batting_runs: 38,
      batting_strike_rate: 271.4,
      boundaries: 6,
      player_form: 85,
      playing_time_percentage: 90
    },
    three_fer: {
      bowling_wickets: 3,
      bowling_economy: 5.25,
      overs_bowled: 4.0,
      bowling_average: 7.0,
      batting_runs: 6,
      player_form: 90
    },
    all_rounder: {
      batting_runs: 42,
      batting_strike_rate: 175.0,
      boundaries: 5,
      bowling_wickets: 2,
      bowling_economy: 6.75,
      overs_bowled: 4.0,
      player_form: 92
    },
    early_dismissal: {
      batting_runs: 2,
      batting_strike_rate: 50.0,
      boundaries: 0,
      bowling_wickets: 0,
      bowling_economy: 12.0,
      overs_bowled: 2.0,
      player_form: 35
    }
  };

  scenarioCards.forEach(card => {
    const scKey = card.getAttribute("data-scenario");
    const applyBtn = card.querySelector(".btn-apply-sc") || card;

    applyBtn.addEventListener("click", (e) => {
      e.stopPropagation();
      if (scenarios[scKey]) {
        populateFormFields(scenarios[scKey]);
        runRealtimePrediction();

        // Visual flash feedback
        card.style.borderColor = "var(--accent-emerald)";
        setTimeout(() => card.style.borderColor = "", 1000);
      }
    });
  });
}

// ==========================================================================
// 7. In-Play Live Match Simulator & Canvas Trajectory
// ==========================================================================
function initLiveMatchSimulator() {
  const playBtn = document.getElementById("simPlayBtn");
  const stepBtn = document.getElementById("simStepBtn");
  const resetBtn = document.getElementById("simResetBtn");
  const scenarioSelect = document.getElementById("simScenarioSelect");

  if (!playBtn || !stepBtn || !resetBtn) return;

  scenarioSelect.addEventListener("change", (e) => {
    simulationState.scenario = e.target.value;
    resetSimulation();
  });

  playBtn.addEventListener("click", () => {
    if (simulationState.active) {
      pauseSimulation();
    } else {
      startSimulation();
    }
  });

  stepBtn.addEventListener("click", () => {
    pauseSimulation();
    stepSimulationOver();
  });

  resetBtn.addEventListener("click", () => {
    resetSimulation();
  });
}

function startSimulation() {
  const playBtn = document.getElementById("simPlayBtn");
  simulationState.active = true;
  if (playBtn) {
    playBtn.innerHTML = `<span>⏸</span> Pause Simulator`;
    playBtn.classList.remove("sim-btn-play");
    playBtn.classList.add("sim-btn-step");
  }

  simulationInterval = setInterval(() => {
    if (simulationState.stepIndex >= simulationState.maxSteps) {
      pauseSimulation();
      appendCommentary("20.0", "🏁 Inning complete! Final fantasy performance evaluated by LightGBM model.", "boundary");
      return;
    }
    stepSimulationOver();
  }, 1200); // 1.2 second per simulated over
}

function pauseSimulation() {
  simulationState.active = false;
  clearInterval(simulationInterval);
  const playBtn = document.getElementById("simPlayBtn");
  if (playBtn) {
    playBtn.innerHTML = `<span>▶</span> Auto Simulate`;
    playBtn.classList.add("sim-btn-play");
    playBtn.classList.remove("sim-btn-step");
  }
}

function resetSimulation() {
  pauseSimulation();
  simulationState.stepIndex = 0;
  simulationState.currentRuns = 0;
  simulationState.currentBalls = 0;
  simulationState.fours = 0;
  simulationState.sixes = 0;
  simulationState.wickets = 0;
  simulationState.overs = 0;
  simulationState.runsConceded = 0;
  simulationState.scoreHistory = [0];

  updateScoreboardUI();
  renderTrajectoryChart();

  const commentaryFeed = document.getElementById("simCommentaryFeed");
  if (commentaryFeed) {
    commentaryFeed.innerHTML = `
      <div class="commentary-item start">
        <span class="com-over">0.0</span>
        <span class="com-text">Match reset. Click <strong>"▶ Auto Simulate"</strong> or <strong>"⏭ Next Over"</strong> to stream real-time match events!</span>
      </div>
    `;
  }
}

function stepSimulationOver() {
  simulationState.stepIndex++;
  const overNum = simulationState.stepIndex;
  
  // Scenario-based stochastic event generator
  let runsThisOver = 0;
  let boundariesThisOver = 0;
  let isSix = false;
  let commentaryText = "";
  let commentaryClass = "";

  const sc = simulationState.scenario;

  if (sc === "chase_top_order") {
    // Steady accumulation
    const rand = Math.random();
    if (rand > 0.7) {
      runsThisOver = 12;
      simulationState.fours += 2;
      boundariesThisOver = 2;
      commentaryText = `Cracking shots! Two consecutive FOURS through the covers (+${runsThisOver} runs).`;
      commentaryClass = "boundary";
    } else if (rand > 0.4) {
      runsThisOver = 8;
      simulationState.fours += 1;
      boundariesThisOver = 1;
      commentaryText = `Well placed into the gap for FOUR! Rotating strike neatly (+${runsThisOver} runs).`;
      commentaryClass = "boundary";
    } else {
      runsThisOver = 4;
      commentaryText = `Smart running between wickets, taking the singles (+${runsThisOver} runs).`;
    }
    simulationState.currentRuns += runsThisOver;
    simulationState.currentBalls += 6;
  } else if (sc === "death_overs") {
    // High-risk explosive
    const rand = Math.random();
    if (rand > 0.5) {
      runsThisOver = 18;
      simulationState.sixes += 2;
      simulationState.fours += 1;
      boundariesThisOver = 3;
      commentaryText = `🚀 MAXIMUM! Clean strike into the top tier! 2x SIXES and a FOUR (+${runsThisOver} runs)!`;
      commentaryClass = "boundary";
    } else {
      runsThisOver = 9;
      simulationState.sixes += 1;
      boundariesThisOver = 1;
      commentaryText = `Dispatched over long-on for a huge SIX! (+${runsThisOver} runs).`;
      commentaryClass = "boundary";
    }
    simulationState.currentRuns += runsThisOver;
    simulationState.currentBalls += 6;
  } else if (sc === "pace_spell") {
    // Strike bowling
    simulationState.overs += 1.0;
    const rand = Math.random();
    if (rand > 0.5 && simulationState.wickets < 4) {
      simulationState.wickets += 1;
      runsThisOver = 4;
      simulationState.runsConceded += runsThisOver;
      commentaryText = `🎯 WICKET! Timber rattled with an express 145km/h yorker! (+1 Wicket, only ${runsThisOver} runs conceded).`;
      commentaryClass = "wicket";
    } else {
      runsThisOver = 6;
      simulationState.runsConceded += runsThisOver;
      commentaryText = `Disciplined length, keeping it tight with dots and singles (${runsThisOver} runs).`;
    }
  } else {
    // All-Round
    simulationState.currentRuns += 7;
    simulationState.currentBalls += 5;
    if (overNum % 2 === 0 && simulationState.wickets < 2) {
      simulationState.wickets += 1;
      simulationState.overs += 1.0;
      simulationState.runsConceded += 6;
      commentaryText = `🔥 Breakthrough! Picks up a crucial WICKET and chipped 7 runs with the bat!`;
      commentaryClass = "wicket";
    } else {
      simulationState.fours += 1;
      boundariesThisOver = 1;
      commentaryText = `Crisp cut shot for FOUR (+7 runs).`;
      commentaryClass = "boundary";
    }
  }

  // Calculate live batting metrics
  const sr = simulationState.currentBalls > 0 
    ? ((simulationState.currentRuns / simulationState.currentBalls) * 100).toFixed(1)
    : "0.0";
  const econ = simulationState.overs > 0 
    ? (simulationState.runsConceded / simulationState.overs).toFixed(2)
    : "0.0";

  // Sync back to Real-Time inputs for live prediction
  populateFormFields({
    batting_runs: simulationState.currentRuns,
    batting_strike_rate: parseFloat(sr) || 120,
    boundaries: simulationState.fours + simulationState.sixes,
    bowling_wickets: simulationState.wickets,
    bowling_economy: parseFloat(econ) || 7.5,
    overs_bowled: simulationState.overs
  });

  // Call prediction
  runRealtimePrediction().then(() => {
    const scoreElem = document.getElementById("predictedScoreDisplay");
    const currentScore = parseFloat(scoreElem ? scoreElem.textContent : "0") || 0;
    simulationState.scoreHistory.push(currentScore);

    updateScoreboardUI(sr, econ);
    renderTrajectoryChart();
    appendCommentary(`${overNum}.0`, `${commentaryText} Projected Score: <strong>${currentScore.toFixed(1)} pts</strong>`, commentaryClass);
  });
}

function updateScoreboardUI(sr = "0.0", econ = "0.0") {
  const runsElem = document.getElementById("simLiveRuns");
  const srElem = document.getElementById("simLiveSR");
  const boundElem = document.getElementById("simLiveBoundaries");
  const bowlElem = document.getElementById("simLiveBowling");
  const matchStatus = document.getElementById("simMatchStatus");
  const currentBadge = document.getElementById("simCurrentScoreBadge");

  if (runsElem) runsElem.textContent = `${simulationState.currentRuns} (${simulationState.currentBalls})`;
  if (srElem) srElem.textContent = sr;
  if (boundElem) boundElem.textContent = `${simulationState.fours} / ${simulationState.sixes}`;
  if (bowlElem) bowlElem.textContent = `${simulationState.wickets} wkt / ${simulationState.overs.toFixed(1)} ov (${econ} econ)`;
  if (matchStatus) matchStatus.textContent = `Live: Over ${simulationState.stepIndex}.0 of 20 • Innings 2`;

  const lastScore = simulationState.scoreHistory[simulationState.scoreHistory.length - 1] || 0;
  if (currentBadge) currentBadge.textContent = `Projected: ${lastScore.toFixed(1)} pts`;
}

function appendCommentary(over, text, typeClass = "") {
  const feed = document.getElementById("simCommentaryFeed");
  if (!feed) return;

  const item = document.createElement("div");
  item.className = `commentary-item ${typeClass}`;
  item.innerHTML = `
    <span class="com-over">${over}</span>
    <span class="com-text">${text}</span>
  `;

  feed.insertBefore(item, feed.firstChild);
}

function renderTrajectoryChart() {
  const canvas = document.getElementById("simTrajectoryCanvas");
  if (!canvas) return;
  const ctx = canvas.getContext("2d");
  const w = canvas.width;
  const h = canvas.height;

  ctx.clearRect(0, 0, w, h);

  const data = simulationState.scoreHistory.length > 0 ? simulationState.scoreHistory : [0];
  const maxVal = Math.max(100, Math.max(...data) + 15);
  const minVal = 0;

  // Background Grid Lines
  ctx.strokeStyle = "rgba(51, 65, 85, 0.3)";
  ctx.lineWidth = 1;
  for (let i = 0; i <= 4; i++) {
    const y = h - 25 - (i / 4) * (h - 45);
    ctx.beginPath();
    ctx.moveTo(40, y);
    ctx.lineTo(w - 20, y);
    ctx.stroke();

    ctx.fillStyle = "#64748b";
    ctx.font = "10px monospace";
    ctx.fillText(`${Math.round(minVal + (i / 4) * maxVal)} pts`, 5, y + 3);
  }

  // Draw Area Gradient & Stroke Line
  if (data.length > 1) {
    const stepX = (w - 70) / Math.max(18, data.length - 1);

    // Gradient fill under curve
    const grad = ctx.createLinearGradient(0, 0, 0, h);
    grad.addColorStop(0, "rgba(6, 182, 212, 0.35)");
    grad.addColorStop(1, "rgba(6, 182, 212, 0.0)");

    ctx.beginPath();
    ctx.moveTo(50, h - 25);

    data.forEach((val, i) => {
      const x = 50 + i * stepX;
      const y = h - 25 - ((val - minVal) / maxVal) * (h - 45);
      if (i === 0) ctx.moveTo(x, y);
      else ctx.lineTo(x, y);
    });

    // Close path for area
    const lastX = 50 + (data.length - 1) * stepX;
    ctx.lineTo(lastX, h - 25);
    ctx.lineTo(50, h - 25);
    ctx.fillStyle = grad;
    ctx.fill();

    // Line Path
    ctx.beginPath();
    ctx.strokeStyle = "#06b6d4";
    ctx.lineWidth = 3;
    ctx.shadowColor = "rgba(6, 182, 212, 0.8)";
    ctx.shadowBlur = 8;

    data.forEach((val, i) => {
      const x = 50 + i * stepX;
      const y = h - 25 - ((val - minVal) / maxVal) * (h - 45);
      if (i === 0) ctx.moveTo(x, y);
      else ctx.lineTo(x, y);
    });
    ctx.stroke();
    ctx.shadowBlur = 0; // reset shadow

    // Points
    data.forEach((val, i) => {
      const x = 50 + i * stepX;
      const y = h - 25 - ((val - minVal) / maxVal) * (h - 45);

      ctx.beginPath();
      ctx.arc(x, y, 4, 0, Math.PI * 2);
      ctx.fillStyle = i === data.length - 1 ? "#10b981" : "#38bdf8";
      ctx.fill();
      ctx.strokeStyle = "#0f172a";
      ctx.lineWidth = 2;
      ctx.stroke();
    });
  }
}

// ==========================================================================
// 8. Leaderboard Search & Role Filter
// ==========================================================================
function initLeaderboardSearch() {
  const searchInput = document.getElementById("rankingsSearch");
  const roleFilter = document.getElementById("roleFilter");
  const table = document.getElementById("rankingsTable");
  const countBadge = document.getElementById("rankingsCountBadge");

  if (!table || !searchInput) return;

  function filterTable() {
    const query = searchInput.value.toLowerCase().trim();
    const selectedRole = roleFilter ? roleFilter.value.toLowerCase() : "";
    const rows = table.querySelectorAll("tbody tr");
    let visibleCount = 0;

    rows.forEach(row => {
      const playerName = row.getAttribute("data-player")?.toLowerCase() || "";
      const team = row.getAttribute("data-team")?.toLowerCase() || "";
      const role = row.getAttribute("data-role")?.toLowerCase() || "";

      const matchesQuery = query === "" || playerName.includes(query) || team.includes(query);
      const matchesRole = selectedRole === "" || role.includes(selectedRole);

      if (matchesQuery && matchesRole) {
        row.style.display = "";
        visibleCount++;
      } else {
        row.style.display = "none";
      }
    });

    if (countBadge) {
      countBadge.textContent = `Showing ${visibleCount} of ${rows.length} Players`;
    }
  }

  searchInput.addEventListener("input", filterTable);
  if (roleFilter) roleFilter.addEventListener("change", filterTable);
}

function initTableSorting() {
  const table = document.getElementById("rankingsTable");
  if (!table) return;

  const headers = table.querySelectorAll("th[data-sort]");
  headers.forEach(header => {
    header.style.cursor = "pointer";
    header.addEventListener("click", () => {
      const sortKey = header.getAttribute("data-sort");
      const isAsc = header.classList.contains("sort-asc");
      
      headers.forEach(h => h.classList.remove("sort-asc", "sort-desc"));
      header.classList.add(isAsc ? "sort-desc" : "sort-asc");

      const tbody = table.querySelector("tbody");
      const rows = Array.from(tbody.querySelectorAll("tr"));

      rows.sort((a, b) => {
        let valA = a.getAttribute(`data-${sortKey}`);
        let valB = b.getAttribute(`data-${sortKey}`);

        let numA = parseFloat(valA);
        let numB = parseFloat(valB);

        if (!isNaN(numA) && !isNaN(numB)) {
          return isAsc ? numB - numA : numA - numB;
        }
        return isAsc ? valB.localeCompare(valA) : valA.localeCompare(valB);
      });

      rows.forEach(row => tbody.appendChild(row));
    });
  });
}

// ==========================================================================
// 10. Squad Optimizer: Team vs Opponent Dynamic Mutual Filtering
// ==========================================================================
function initSquadOptimizerFilters() {
  const optTeam1 = document.getElementById("optTeam1");
  const optTeam2 = document.getElementById("optTeam2");
  if (!optTeam1 || !optTeam2) return;

  // Retrieve master team list from data attribute or window
  let allTeams = [];
  try {
    if (optTeam2.dataset.allTeams) {
      allTeams = JSON.parse(optTeam2.dataset.allTeams);
    }
  } catch (e) {}

  if (!allTeams || allTeams.length === 0) {
    if (window.SQUAD_TEAMS && Array.isArray(window.SQUAD_TEAMS)) {
      allTeams = window.SQUAD_TEAMS;
    } else {
      allTeams = Array.from(optTeam2.options)
        .map(o => o.value.trim())
        .filter(v => v !== "");
    }
  }

  function update() {
    const selectedTeam1 = (optTeam1.value || "").trim();
    const currentTeam2 = (optTeam2.value || "").trim();

    // Rebuild Opponent dropdown
    optTeam2.innerHTML = '<option value="">All Teams (League Dream Team)</option>';

    allTeams.forEach(team => {
      const cleanTeam = (team || "").trim();
      if (!cleanTeam) return;

      // When Team 1 is selected, it must NOT appear in Opponent dropdown
      if (selectedTeam1 && cleanTeam.toLowerCase() === selectedTeam1.toLowerCase()) {
        return;
      }

      const opt = document.createElement("option");
      opt.value = cleanTeam;
      opt.textContent = cleanTeam;
      if (currentTeam2 && currentTeam2.toLowerCase() === cleanTeam.toLowerCase()) {
        opt.selected = true;
      }
      optTeam2.appendChild(opt);
    });

    // Reset Opponent if it previously had the team now chosen as Team 1
    if (selectedTeam1 && currentTeam2.toLowerCase() === selectedTeam1.toLowerCase()) {
      optTeam2.value = "";
    }
  }

  optTeam1.addEventListener("change", update);
  update();
  window.addEventListener("pageshow", update);
}
