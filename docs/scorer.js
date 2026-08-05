/* VALID self-scorer.
   No network calls, no storage, no analytics: the answers stay in this tab.
   Numbers quoted below come from the paper (KDD-MLF 2026 / SSRN 6508779):
   median 2.5 of 12 across 74 empirical papers; AUC-alone false-positive rate
   27% [21, 34] falling to 0% [0, 1.9] once CPCV with PBO is added. */
(function () {
  "use strict";

  var MEDIAN = 2.5;
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
    if (c.total < 4) {
      return "Most gaps concentrate in Stage " + weakStage(c) +
             " — the discovery review starts there.";
    }
    if (c.total <= 8) {
      return "You're above the published median. The remaining items are " +
             "exactly what an audit pins down against your artifacts.";
    }
    return "Strong. The difference between self-scoring and an audit is " +
           "evidence — have it verified.";
  }

  function verdict(c) {
    if (c.total === 0) return "Check the items your study satisfies.";

    if (c.total < 4) {
      return "At or below the median of the 74 empirical papers audited " +
             "(2.5 of 12). The items left unchecked are the ones the audit " +
             "found missing most often.";
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
      return "Above the audit median. " + gap;
    }
    return "Every item you scored is satisfied. Note what this is: a " +
           "self-assessment of your own reading of your own artifacts. " +
           "We scored our own paper 9 of 12 and disclosed three partials.";
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
      "Your score of " + c.total + " out of 12, against a median of 2.5 out " +
      "of 12 in an audit of 74 empirical papers."
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
           "Median in an audit of 74 empirical papers: " + MEDIAN + "/12. " +
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
