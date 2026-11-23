/-!
# LabDecideLean: Formal L6 Conformal Calibration & Stepped Frontier Trust Regime

Self-contained Lean 4 (`v4.36.0`) formalization for `lab-decide` (`L6`), synchronized
with the central 14-repository Auto-Lab Trust Regime (`AutoLabLean`):

1. **Finite-Sample Split Conformal Calibration & kNN Trust Boundary (`ConformalQuery`, `checkDecide`)**:
   - Exact finite-sample order-statistic rank:
     `conformalRank n alphaPermille = ceil((n + 1) * (1000 - alphaPermille) / 1000)`.
   - Never silently clamps `k > n`: undersized calibration sets MUST abstain (`undersized_calib_always_abstains`).
   - Never extrapolates outside the calibrated kNN support (`out_of_domain_always_abstains`).
   - Enforces ordered intervals within physical solubility bounds `[physMinMilli, physMaxMilli]`.
2. **Stepped Annular Frontier Calibration (`FrontierProbeSpec`, `checkFrontierAction`)**:
   - Resolves the exploration-vs-safety paradox by allowing micro-volume (`<= 10.0 uL`),
     zero-synthesis (`isDirectBuy = true`), calibration-only stepping-stone probes
     inside an annular frontier shell `tau < d(x) <= (1 + eta) * tau` while blocking
     all deep-OOD leaps (`deep_ood_always_blocked`, `accepted_outside_tau_is_bounded_probe`).
-/
namespace LabDecideLean

/-- Structured boundary violation reporting the repository, rule identifier, and detail. -/
structure RegimeViolation where
  repo   : String
  rule   : String
  detail : String
  deriving DecidableEq, Repr

def RegimeViolation.format (v : RegimeViolation) : String :=
  s!"[{v.repo} :: {v.rule}] {v.detail}"

/-- Return the first violation in a list of `(holds, violation)` checks. -/
def firstViolation : List (Bool × RegimeViolation) → Option RegimeViolation
  | [] => none
  | (true, _) :: rest => firstViolation rest
  | (false, v) :: _ => some v

@[simp] theorem firstViolation_nil : firstViolation [] = none := rfl

@[simp] theorem firstViolation_cons (b : Bool) (v : RegimeViolation)
    (rest : List (Bool × RegimeViolation)) :
    firstViolation ((b, v) :: rest) = none ↔ b = true ∧ firstViolation rest = none := by
  cases b <;> simp [firstViolation]

theorem firstViolation_none_iff (checks : List (Bool × RegimeViolation)) :
    firstViolation checks = none ↔ ∀ p ∈ checks, p.1 = true := by
  induction checks with
  | nil => simp [firstViolation]
  | cons hd tl ih =>
    rcases hd with ⟨b, v⟩
    cases b with
    | false => simp [firstViolation]
    | true => simp [firstViolation, ih]

/-! ## 1. Finite-Sample Conformal Calibration & Applicability-Domain Gate -/

/-- Exact integer ceiling `ceil((n + 1) * (1000 - alphaPermille) / 1000)`. -/
def conformalRank (n alphaPermille : Nat) : Nat :=
  ((n + 1) * (1000 - alphaPermille) + 999) / 1000

/-- Fixed-point summary of a conformal interval and applicability-domain query. -/
structure ConformalQuery where
  nCalib           : Nat
  alphaPermille    : Nat
  knnDistPermille  : Nat
  tauDistPermille  : Nat
  loMilli          : Int
  hiMilli          : Int
  physMinMilli     : Int
  physMaxMilli     : Int
  deriving DecidableEq, Repr

/-- Declarative specification of a valid, in-domain conformal decision. -/
structure ConformalDecisionOK (q : ConformalQuery) : Prop where
  alphaValid      : 0 < q.alphaPermille ∧ q.alphaPermille < 1000
  calibNonEmpty   : 0 < q.nCalib
  rankWithinCalib : conformalRank q.nCalib q.alphaPermille ≤ q.nCalib
  insideTrustZone : q.knnDistPermille ≤ q.tauDistPermille
  intervalOrdered : q.loMilli ≤ q.hiMilli
  withinPhysMin   : q.physMinMilli ≤ q.loMilli
  withinPhysMax   : q.hiMilli ≤ q.physMaxMilli

/-- Decidable guardrail checker for `lab-decide` conformal predictions. -/
def checkDecide (q : ConformalQuery) : Option RegimeViolation :=
  firstViolation
    [ (decide (0 < q.alphaPermille ∧ q.alphaPermille < 1000),
        { repo := "lab-decide", rule := "alpha_in_open_unit_interval",
          detail := "alphaPermille must lie strictly in (0, 1000)" }),
      (decide (0 < q.nCalib),
        { repo := "lab-decide", rule := "nonempty_calibration_set",
          detail := "calibration set size nCalib must be positive" }),
      (decide (conformalRank q.nCalib q.alphaPermille ≤ q.nCalib),
        { repo := "lab-decide", rule := "finite_sample_conformal_rank",
          detail := "ceil((n+1)*(1-alpha)) exceeds nCalib; interval is infinite -> must ABSTAIN" }),
      (decide (q.knnDistPermille ≤ q.tauDistPermille),
        { repo := "lab-decide", rule := "knn_trust_boundary",
          detail := "candidate distance exceeds calibrated kNN trust threshold -> must ABSTAIN" }),
      (decide (q.loMilli ≤ q.hiMilli),
        { repo := "lab-decide", rule := "interval_well_ordered",
          detail := "lower conformal bound exceeds upper bound" }),
      (decide (q.physMinMilli ≤ q.loMilli ∧ q.hiMilli ≤ q.physMaxMilli),
        { repo := "lab-decide", rule := "physical_solubility_bounds",
          detail := "prediction interval violates physical solubility range [physMin, physMax]" }) ]

/-- Soundness and completeness of `checkDecide` wrt `ConformalDecisionOK`. -/
theorem checkDecide_ok_iff (q : ConformalQuery) :
    checkDecide q = none ↔ ConformalDecisionOK q := by
  unfold checkDecide
  simp only [firstViolation_cons, firstViolation_nil, and_true, decide_eq_true_eq]
  constructor
  · rintro ⟨h1, h2, h3, h4, h5, h6a, h6b⟩
    exact ⟨h1, h2, h3, h4, h5, h6a, h6b⟩
  · rintro ⟨h1, h2, h3, h4, h5, h6a, h6b⟩
    exact ⟨h1, h2, h3, h4, h5, h6a, h6b⟩

/-- Hard boundary theorem 1: Out-of-domain molecules (`knnDist > tauDist`) are never accepted. -/
theorem out_of_domain_always_abstains (q : ConformalQuery)
    (hood : q.tauDistPermille < q.knnDistPermille) :
    checkDecide q ≠ none := by
  intro hok
  have hd := ((checkDecide_ok_iff q).mp hok).insideTrustZone
  omega

/-- Hard boundary theorem 2: Undersized calibration sets (`k > n`) are never silently clamped. -/
theorem undersized_calib_always_abstains (q : ConformalQuery)
    (hsmall : q.nCalib < conformalRank q.nCalib q.alphaPermille) :
    checkDecide q ≠ none := by
  intro hok
  have hr := ((checkDecide_ok_iff q).mp hok).rankWithinCalib
  omega

/-- Whenever `alphaPermille < 1000`, the finite-sample conformal rank is strictly `>= 1`. -/
theorem conformalRank_pos (n alphaPermille : Nat) (ha : alphaPermille < 1000) :
    1 ≤ conformalRank n alphaPermille := by
  unfold conformalRank
  have hpos : 0 < (n + 1) * (1000 - alphaPermille) :=
    Nat.mul_pos (Nat.succ_pos n) (by omega)
  omega

/-- Accepted conformal intervals always lie ordered within `[physMinMilli, physMaxMilli]`. -/
theorem physical_bounds_respected (q : ConformalQuery)
    (hok : checkDecide q = none) :
    q.physMinMilli ≤ q.loMilli ∧ q.loMilli ≤ q.hiMilli ∧ q.hiMilli ≤ q.physMaxMilli := by
  have h := (checkDecide_ok_iff q).mp hok
  exact ⟨h.withinPhysMin, h.intervalOrdered, h.withinPhysMax⟩

/-! ## 2. Stepped Annular Frontier Calibration (`Chem-RRT` Stepping-Stone Guard) -/

inductive CampaignMode where
  | exploitation
  | frontierCalibration
  deriving DecidableEq, Repr

/-- Specification of an exploitation or annular stepping-stone frontier action. -/
structure FrontierProbeSpec where
  mode                  : CampaignMode
  knnDistPermille       : Nat
  tauPermille           : Nat
  etaStepPermille       : Nat
  maxEtaPermille        : Nat
  assayVolTenthUL       : Nat
  microVolCapTenthUL    : Nat
  isDirectBuy           : Bool
  usedAsPointPrediction : Bool
  deriving DecidableEq, Repr

/-- Outer radius `tau + floor(tau * etaPermille / 1000)` of the stepping-stone shell. -/
def frontierOuterRadius (tau etaPermille : Nat) : Nat :=
  tau + (tau * etaPermille) / 1000

theorem frontierOuterRadius_ge_tau (tau etaPermille : Nat) :
    tau ≤ frontierOuterRadius tau etaPermille := by
  unfold frontierOuterRadius
  omega

def checkFrontierAction (p : FrontierProbeSpec) : Option RegimeViolation :=
  match p.mode with
  | .exploitation =>
    firstViolation
      [ (decide (p.knnDistPermille ≤ p.tauPermille),
          { repo := "lab-decide", rule := "exploitation_inside_tau",
            detail := "exploitation run exceeds current calibrated trust radius tau" }) ]
  | .frontierCalibration =>
    firstViolation
      [ (decide (p.etaStepPermille ≤ p.maxEtaPermille),
          { repo := "lab-decide", rule := "bounded_annular_shell_width",
            detail := "frontier stepping-stone expansion ratio eta exceeds safe shell cap" }),
        (decide (p.knnDistPermille ≤ frontierOuterRadius p.tauPermille p.etaStepPermille),
          { repo := "lab-decide", rule := "within_stepping_stone_shell",
            detail := "candidate lies in deep OOD beyond the (1 + eta) * tau stepping-stone shell" }),
        (decide (p.assayVolTenthUL ≤ p.microVolCapTenthUL),
          { repo := "lab-rehearsal", rule := "frontier_micro_volume_cap",
            detail := "frontier calibration probe exceeds micro-scale reagent volume cap" }),
        (p.isDirectBuy,
          { repo := "buy-or-make", rule := "frontier_requires_direct_stock",
            detail := "multi-step synthesis forbidden outside calibrated trust zone; probe stock leaves first" }),
        (!p.usedAsPointPrediction,
          { repo := "lab-decide", rule := "frontier_calibration_only",
            detail := "frontier probe cannot be consumed as a calibrated point prediction prior to assay" }) ]

/-- Deep OOD candidates beyond `(1 + eta) * tau` are blocked in ALL campaign modes. -/
theorem deep_ood_always_blocked (p : FrontierProbeSpec)
    (hdeep : frontierOuterRadius p.tauPermille p.etaStepPermille < p.knnDistPermille) :
    checkFrontierAction p ≠ none := by
  intro hok
  cases hm : p.mode with
  | exploitation =>
    simp only [checkFrontierAction, hm, firstViolation_cons, firstViolation_nil,
               and_true, decide_eq_true_eq] at hok
    unfold frontierOuterRadius at hdeep
    omega
  | frontierCalibration =>
    simp only [checkFrontierAction, hm, firstViolation_cons, firstViolation_nil,
               and_true, decide_eq_true_eq] at hok
    rcases hok with ⟨_, hshell, _, _, _⟩
    omega

/-- Any candidate accepted outside `tau` is provably a micro-scale, direct-buy calibration probe. -/
theorem accepted_outside_tau_is_bounded_probe (p : FrontierProbeSpec)
    (hout : p.tauPermille < p.knnDistPermille)
    (hok : checkFrontierAction p = none) :
    p.mode = .frontierCalibration ∧
    p.knnDistPermille ≤ frontierOuterRadius p.tauPermille p.etaStepPermille ∧
    p.assayVolTenthUL ≤ p.microVolCapTenthUL ∧
    p.isDirectBuy = true ∧
    p.usedAsPointPrediction = false := by
  cases hm : p.mode with
  | exploitation =>
    simp only [checkFrontierAction, hm, firstViolation_cons, firstViolation_nil,
               and_true, decide_eq_true_eq] at hok
    omega
  | frontierCalibration =>
    simp only [checkFrontierAction, hm, firstViolation_cons, firstViolation_nil,
               and_true, decide_eq_true_eq, Bool.not_eq_true'] at hok
    rcases hok with ⟨_, hshell, hvol, hbuy, hcal⟩
    exact ⟨rfl, hshell, hvol, hbuy, hcal⟩

/-! ## 3. Compile-Time `#guard` Witnesses -/

#guard (checkDecide
         { nCalib := 180, alphaPermille := 100, knnDistPermille := 410,
           tauDistPermille := 758, loMilli := -2410, hiMilli := 630,
           physMinMilli := -15000, physMaxMilli := 1750 }).isNone == true

-- Undersized calibration set (`n = 5` at `alpha = 0.10` -> `rank = 6 > 5`) -> ABSTAIN
#guard (checkDecide
         { nCalib := 5, alphaPermille := 100, knnDistPermille := 300,
           tauDistPermille := 758, loMilli := -2000, hiMilli := 500,
           physMinMilli := -15000, physMaxMilli := 1750 }).isSome == true

-- Out-of-domain candidate (`knnDist = 1380 > 758`) -> ABSTAIN
#guard (checkDecide
         { nCalib := 180, alphaPermille := 100, knnDistPermille := 1380,
           tauDistPermille := 758, loMilli := -2410, hiMilli := 630,
           physMinMilli := -15000, physMaxMilli := 1750 }).isSome == true

-- Valid annular stepping-stone probe (`758 < 900 <= 947`, `8.0 uL <= 10.0 uL`, direct buy) -> OK
#guard (checkFrontierAction
         { mode := .frontierCalibration, knnDistPermille := 900, tauPermille := 758,
           etaStepPermille := 250, maxEtaPermille := 350, assayVolTenthUL := 80,
           microVolCapTenthUL := 100, isDirectBuy := true,
           usedAsPointPrediction := false }).isNone == true

-- Deep OOD leap (`1400 > 947`) -> BLOCKED
#guard (checkFrontierAction
         { mode := .frontierCalibration, knnDistPermille := 1400, tauPermille := 758,
           etaStepPermille := 250, maxEtaPermille := 350, assayVolTenthUL := 80,
           microVolCapTenthUL := 100, isDirectBuy := true,
           usedAsPointPrediction := false }).isSome == true

-- Production-volume (`25.0 uL > 10.0 uL`) outside tau -> BLOCKED
#guard (checkFrontierAction
         { mode := .frontierCalibration, knnDistPermille := 900, tauPermille := 758,
           etaStepPermille := 250, maxEtaPermille := 350, assayVolTenthUL := 250,
           microVolCapTenthUL := 100, isDirectBuy := true,
           usedAsPointPrediction := false }).isSome == true

end LabDecideLean
