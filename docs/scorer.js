/* VALID self-scorer.
   No network calls, no storage, no analytics: the answers stay in this tab.
   October 2026: the paper's literature-audit median and its CPCV/PBO
   false-positive result were withdrawn (CHANGELOG_CR.md, E13), so the scorer
   no longer compares a score with a published median. */
(function () {
  "use strict";

  var TOTAL = 12;
  var URL = "https://orcajae.github.io/valid-framework/";

  var boxes = Array.prototype.slice.call(
    document.querySelectorAll('#valid-form input[type="checkbox"]')
  );
  var elTotal = document.getElementById("total");
  var elS1 = document.getElementById("s1");
  var elS2 = document.getElementById("s2");
  var elFill = document.getElementById("bar-fill");
  var elBarImg = document.getElementById("bar-img");
  var elVerdict = document.getElementById("verdict");
  var elV4 = document.getElementById("v4-flag");
  var elCopy = document.getElementById("copy");
  var elStatus = document.getElementById("copy-status");
  var elNext = document.getElementById("next-step");
  var v4 = document.getElementById("v4");

  function counts() {
    var s1 = 0, s2 = 0;
    boxes.forEach(function (b) {
      if (!b.checked) return;
      if (b.getAttribute("data-stage") === "1") s1++; else s2++;
    });
    return { s1: s1, s2: s2, total: s1 + s2 };
  }

  /* Which stage holds more unchecked items. V1-V6 are Stage 1 (statistical),
     V7-V12 Stage 2 (economic), six items each, per Table 1 of the paper.
     Ties resolve to Stage 1: it is verifiable from the manuscript alone. */
  function weakStage(c) {
    var miss1 = 6 - c.s1, miss2 = 6 - c.s2;
    return miss2 > miss1 ? "2" : "1";
  }

  function nextStep(c) {
    if (c.total === 0) return "";
    if (c.total < TOTAL) {
      return "Most of the unchecked items are in Stage " + weakStage(c) +
             ". Start there.";
    }
    return "Check each item against the artifact itself — the code, the " +
           "data and the stored outputs — not against your memory of them.";
  }

  function verdict(c) {
    if (c.total === 0) return "Check the items your study satisfies.";

    if (c.total < 4) {
      return "Most items are unchecked. Each one is a way a backtest can " +
             "look better than the strategy is.";
    }
    if (c.total <= 8) {
      var gap;
      if (c.s1 < c.s2) {
        gap = "The statistical gate is where the gaps remain: Stage 1 items " +
              "are verifiable from the manuscript alone, so reviewers can " +
              "check them without your data.";
      } else if (c.s2 < c.s1) {
        gap = "The economic gate is where the gaps remain: a signal that " +
              "clears Stage 1 can still be unexploitable after costs.";
      } else {
        gap = "The gaps are spread evenly across both gates.";
      }
      return gap;
    }
    return "Every item you scored is satisfied. Note what this is: a " +
           "self-assessment of your own reading of your own artifacts. " +
           "Our own paper's self-assessment did not survive a check of the " +
           "code behind it.";
  }

  function render() {
    var c = counts();
    elTotal.textContent = String(c.total);
    elS1.textContent = String(c.s1);
    elS2.textContent = String(c.s2);

    var pct = (c.total / TOTAL) * 100;
    elFill.style.width = pct + "%";
    elBarImg.setAttribute(
      "aria-label",
      "Your score of " + c.total + " out of 12."
    );

    elVerdict.textContent = verdict(c);
    elV4.hidden = v4.checked;

    var step = nextStep(c);
    elNext.textContent = step;
    elNext.hidden = step === "";

    elStatus.textContent = "";
  }

  function resultText() {
    var c = counts();
    return "VALID self-score: " + c.total + "/12 " +
           "(Stage 1 statistical " + c.s1 + "/6, " +
           "Stage 2 economic " + c.s2 + "/6). " +
           URL;
  }

  function fallbackCopy(text) {
    var ta = document.createElement("textarea");
    ta.value = text;
    ta.setAttribute("readonly", "");
    ta.style.position = "fixed";
    ta.style.top = "-1000px";
    document.body.appendChild(ta);
    ta.select();
    var ok = false;
    try { ok = document.execCommand("copy"); } catch (e) { ok = false; }
    document.body.removeChild(ta);
    return ok;
  }

  function copy() {
    var text = resultText();
    function done(ok) {
      elStatus.textContent = ok
        ? "Copied to the clipboard."
        : "Copy failed — select the text above instead.";
    }
    if (navigator.clipboard && navigator.clipboard.writeText) {
      navigator.clipboard.writeText(text).then(
        function () { done(true); },
        function () { done(fallbackCopy(text)); }
      );
    } else {
      done(fallbackCopy(text));
    }
  }

  boxes.forEach(function (b) { b.addEventListener("change", render); });
  elCopy.addEventListener("click", copy);
  render();
})();
