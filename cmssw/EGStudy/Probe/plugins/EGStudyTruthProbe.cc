// Study probe, not for integration. Truth content of the HGCAL e/gamma objects at PU200.
// Truth units: the reconstructed energy of a cell is shared by the simulated energies of its calo-boundary particles.
// A truth object is the reconstructable-final-state ancestor of a calo-boundary particle (the particle itself when it
// has none). The deposit of a truth object in a reco object is the shared cell energy summed over the hits of the
// reco object, with the hit fractions applied.
// Tree obj: one row per reco object. kind 0 supercluster (coll = position in superClusters), 1 photon, 2 GsfElectron,
// 3 TICL candidate, 4 e/gamma trackster. Common columns: kinematics, the truth deposits and the dominant truth object.
// The x array holds the kind-specific columns (see fillX* below for the order).
// Tree gun: one row per prompt final-state electron or photon of the signal with 1.5 < |eta| < 3, with the graph index
// of its truth object (key) and its HGCAL deposit.

#include <algorithm>
#include <array>
#include <cmath>
#include <string>
#include <unordered_map>
#include <vector>

#include "CommonTools/UtilAlgos/interface/TFileService.h"
#include "DataFormats/CaloRecHit/interface/CaloCluster.h"
#include "DataFormats/Common/interface/MultiSpan.h"
#include "DataFormats/Common/interface/RefProdVector.h"
#include "DataFormats/DetId/interface/DetId.h"
#include "DataFormats/EgammaCandidates/interface/GsfElectron.h"
#include "DataFormats/EgammaCandidates/interface/Photon.h"
#include "DataFormats/EgammaReco/interface/SuperCluster.h"
#include "DataFormats/GsfTrackReco/interface/GsfTrack.h"
#include "DataFormats/HGCRecHit/interface/HGCRecHitCollections.h"
#include "DataFormats/HGCalReco/interface/TICLCandidate.h"
#include "DataFormats/HGCalReco/interface/TICLLayerTile.h"
#include "DataFormats/HGCalReco/interface/Trackster.h"
#include "DataFormats/HepMCCandidate/interface/GenParticle.h"
#include "DataFormats/Math/interface/deltaPhi.h"
#include "DataFormats/Math/interface/deltaR.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/Run.h"
#include "FWCore/Framework/interface/one/EDAnalyzer.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "Geometry/Records/interface/CaloGeometryRecord.h"
#include "RecoLocalCalo/HGCalRecAlgos/interface/TICLGeomTools.h"
#include "RecoTICL/ParticleFlow/interface/EGammaSuperClusterFeatures.h"
#include "SimDataFormats/TruthInfo/interface/Graph.h"
#include "SimDataFormats/TruthInfo/interface/LogicalGraphHitIndex.h"
#include "SimDataFormats/TruthInfo/interface/Particle.h"

#include "TTree.h"

namespace {
  bool isHGCal(uint32_t rawId) {
    const auto det = DetId(rawId).det();
    return det == DetId::HGCalEE || det == DetId::HGCalHSi || det == DetId::HGCalHSc;
  }
  // 0 EM (photon, electron, pi0, eta), 1 charged hadron, 2 neutral hadron, 3 muon, 4 other.
  int speciesOfPdg(int pdg) {
    const int a = std::abs(pdg);
    if (a == 22 || a == 11 || a == 111 || a == 221)
      return 0;
    if (a == 211 || a == 321 || a == 2212 || a == 3222 || a == 3112 || a == 3312 || a == 3334)
      return 1;
    if (a == 130 || a == 310 || a == 2112 || a == 3122 || a == 3212 || a == 3322)
      return 2;
    if (a == 13)
      return 3;
    return 4;
  }
  constexpr int kNX = 48;

  // Sum of E sin(theta) of the layer clusters within dR of (eta, phi), from the TICL layer tiles.
  double isoSumEt(
      const TICLLayerTiles &tiles, const std::vector<reco::CaloCluster> &lcs, float eta, float phi, float dR) {
    double s = 0.;
    for (int layer = 0; layer < ticl::TileConstants::nLayers; ++layer) {
      const auto &tile = tiles[layer];
      const auto box = tile.searchBoxEtaPhi(eta - dR, eta + dR, phi - dR, phi + dR);
      for (int iEta = box[0]; iEta <= box[1]; ++iEta)
        for (int iPhi = box[2]; iPhi <= box[3]; ++iPhi)
          for (auto i : tile[tile.globalBin(iEta, iPhi % ticl::TileConstants::nPhiBins)]) {
            const auto &lc = lcs[i];
            if (reco::deltaR2(lc.eta(), lc.phi(), eta, phi) < dR * dR)
              s += lc.energy() / std::cosh(lc.eta());
          }
    }
    return s;
  }
}  // namespace

class EGStudyTruthProbe : public edm::one::EDAnalyzer<edm::one::SharedResources, edm::one::WatchRuns> {
public:
  explicit EGStudyTruthProbe(const edm::ParameterSet &);
  void beginRun(const edm::Run &, const edm::EventSetup &) override;
  void endRun(const edm::Run &, const edm::EventSetup &) override {}
  void analyze(const edm::Event &, const edm::EventSetup &) override;

private:
  struct Truth {
    float recoE = 0.f, dep = 0.f, sigDep = 0.f, emDep = 0.f, chDep = 0.f, nhDep = 0.f, puEMLead = 0.f, puHadLead = 0.f;
    int domKey = -1, domPdg = 0, domMotherPdg = 0, domSignal = -1, nObj10 = 0, sigKey = -1;
    float domE = 0.f, domEta = 0.f, domPhi = 0.f, domDep = 0.f, domTot = 0.f, sigObjDep = 0.f, sigObjTot = 0.f;
  };
  using Hits = std::vector<std::pair<uint32_t, float>>;
  Truth truthOf(const Hits &hits) const;
  void fillRow(int kind, int coll, int idx, float pt, float eta, float phi, float e, float rawE, const Truth &t);
  double trackIso(const std::vector<reco::Track> &tracks, float eta, float phi) const;

  const edm::EDGetTokenT<std::vector<reco::CaloCluster>> lcToken_;
  const edm::EDGetTokenT<truth::Graph> graphToken_;
  const edm::EDGetTokenT<truth::LogicalGraphHitIndex> hitIndexToken_;
  const edm::EDGetTokenT<std::vector<reco::GenParticle>> genToken_;
  const edm::EDGetTokenT<std::vector<reco::Track>> tracksToken_;
  const edm::EDGetTokenT<TICLLayerTiles> tilesToken_;
  std::vector<edm::EDGetTokenT<HGCRecHitCollection>> recHitTokens_;
  std::vector<edm::EDGetTokenT<std::vector<reco::SuperCluster>>> scTokens_;
  std::vector<edm::EDGetTokenT<std::vector<int>>> scIndexTokens_;
  edm::EDGetTokenT<std::vector<reco::Photon>> photonToken_;
  edm::EDGetTokenT<std::vector<reco::GsfElectron>> electronToken_;
  edm::EDGetTokenT<std::vector<TICLCandidate>> candToken_, candAltToken_;
  edm::EDGetTokenT<std::vector<ticl::Trackster>> egTsToken_;
  edm::EDGetTokenT<std::vector<int>> egIndexToken_, trackToTracksterToken_, trackModeToken_, neutralIdxToken_,
      neutralPdgToken_;
  const edm::EDGetTokenT<std::unordered_map<DetId, const unsigned int>> hitMapToken_;
  const edm::EDGetTokenT<edm::RefProdVector<HGCRecHitCollection>> hitsToken_;
  const edm::ESGetToken<TICLGeomHost, CaloGeometryRecord> ticlGeomToken_;
  const edm::ESGetToken<TICLGeomLookupHost, CaloGeometryRecord> ticlGeomLookupToken_;
  const edm::ESGetToken<TICLGeomLayersHost, CaloGeometryRecord> ticlGeomLayersToken_;
  ticlgeom::Tools tools_;
  const bool useEGammaTracksters_, useCandAlt_;
  const double minObjectPt_;

  // Per-event truth tables.
  const truth::Graph *graph_ = nullptr;
  std::unordered_map<uint32_t, std::vector<std::pair<int, float>>> cellTruth_;
  std::unordered_map<uint32_t, float> recHitEnergy_;
  std::unordered_map<int, int> cbObj_;
  std::unordered_map<int, double> objDep_;
  std::vector<int> objectCache_;

  TTree *objT_, *gunT_;
  unsigned int run_, lumi_;
  unsigned long long event_;
  int kind_, coll_, idx_;
  float pt_, eta_, phi_, e_, rawE_;
  Truth t_;
  float x_[kNX];
  int gPdg_, gKey_;
  float gPt_, gEta_, gPhi_, gE_, gDep_;
};

EGStudyTruthProbe::EGStudyTruthProbe(const edm::ParameterSet &cfg)
    : lcToken_(consumes(cfg.getParameter<edm::InputTag>("layerClusters"))),
      graphToken_(consumes(cfg.getParameter<edm::InputTag>("graph"))),
      hitIndexToken_(consumes(cfg.getParameter<edm::InputTag>("hitIndex"))),
      genToken_(consumes(cfg.getParameter<edm::InputTag>("genParticles"))),
      tracksToken_(consumes(cfg.getParameter<edm::InputTag>("tracks"))),
      tilesToken_(consumes(cfg.getParameter<edm::InputTag>("layerTiles"))),
      hitMapToken_(consumes(cfg.getParameter<edm::InputTag>("hitMap"))),
      hitsToken_(consumes(cfg.getParameter<edm::InputTag>("hits"))),
      ticlGeomToken_(esConsumes<TICLGeomHost, CaloGeometryRecord, edm::Transition::BeginRun>()),
      ticlGeomLookupToken_(esConsumes<TICLGeomLookupHost, CaloGeometryRecord, edm::Transition::BeginRun>()),
      ticlGeomLayersToken_(esConsumes<TICLGeomLayersHost, CaloGeometryRecord, edm::Transition::BeginRun>()),
      useEGammaTracksters_(!cfg.getParameter<edm::InputTag>("eGammaTracksters").label().empty()),
      useCandAlt_(!cfg.getParameter<edm::InputTag>("candidatesAlt").label().empty()),
      minObjectPt_(cfg.getParameter<double>("minObjectPt")) {
  usesResource(TFileService::kSharedResource);
  for (auto const &tag : cfg.getParameter<std::vector<edm::InputTag>>("recHits"))
    recHitTokens_.push_back(consumes<HGCRecHitCollection>(tag));
  for (auto const &tag : cfg.getParameter<std::vector<edm::InputTag>>("superClusters")) {
    scTokens_.push_back(consumes<std::vector<reco::SuperCluster>>(tag));
    scIndexTokens_.push_back(mayConsume<std::vector<int>>(edm::InputTag(tag.label(), "tracksterIndex", tag.process())));
  }
  photonToken_ = consumes(cfg.getParameter<edm::InputTag>("photons"));
  electronToken_ = consumes(cfg.getParameter<edm::InputTag>("electrons"));
  candToken_ = consumes(cfg.getParameter<edm::InputTag>("candidates"));
  if (useCandAlt_)
    candAltToken_ = consumes(cfg.getParameter<edm::InputTag>("candidatesAlt"));
  if (useEGammaTracksters_) {
    const auto tag = cfg.getParameter<edm::InputTag>("eGammaTracksters");
    egTsToken_ = consumes(tag);
    egIndexToken_ = consumes(edm::InputTag(tag.label(), "interpretationIndex", tag.process()));
    const auto it = cfg.getParameter<std::string>("interpretations");
    trackToTracksterToken_ = consumes(edm::InputTag(it, "trackToTrackster"));
    trackModeToken_ = consumes(edm::InputTag(it, "trackMode"));
    neutralIdxToken_ = consumes(edm::InputTag(it, "neutralIdx"));
    neutralPdgToken_ = consumes(edm::InputTag(it, "neutralPdg"));
  }
  edm::Service<TFileService> fs;
  objT_ = fs->make<TTree>("obj", "e/gamma objects with truth");
  objT_->Branch("run", &run_);
  objT_->Branch("lumi", &lumi_);
  objT_->Branch("event", &event_);
  for (auto [n, p] : std::vector<std::pair<const char *, int *>>{{"kind", &kind_},
                                                                 {"coll", &coll_},
                                                                 {"idx", &idx_},
                                                                 {"domKey", &t_.domKey},
                                                                 {"domPdg", &t_.domPdg},
                                                                 {"domMotherPdg", &t_.domMotherPdg},
                                                                 {"domSignal", &t_.domSignal},
                                                                 {"nObj10", &t_.nObj10},
                                                                 {"sigKey", &t_.sigKey}})
    objT_->Branch(n, p);
  for (auto [n, p] : std::vector<std::pair<const char *, float *>>{{"pt", &pt_},
                                                                   {"eta", &eta_},
                                                                   {"phi", &phi_},
                                                                   {"e", &e_},
                                                                   {"rawE", &rawE_},
                                                                   {"recoE", &t_.recoE},
                                                                   {"dep", &t_.dep},
                                                                   {"sigDep", &t_.sigDep},
                                                                   {"emDep", &t_.emDep},
                                                                   {"chDep", &t_.chDep},
                                                                   {"nhDep", &t_.nhDep},
                                                                   {"puEMLead", &t_.puEMLead},
                                                                   {"puHadLead", &t_.puHadLead},
                                                                   {"domE", &t_.domE},
                                                                   {"domEta", &t_.domEta},
                                                                   {"domPhi", &t_.domPhi},
                                                                   {"domDep", &t_.domDep},
                                                                   {"domTot", &t_.domTot},
                                                                   {"sigObjDep", &t_.sigObjDep},
                                                                   {"sigObjTot", &t_.sigObjTot}})
    objT_->Branch(n, p);
  objT_->Branch("x", x_, "x[48]/F");
  gunT_ = fs->make<TTree>("gun", "prompt signal electrons and photons");
  gunT_->Branch("run", &run_);
  gunT_->Branch("lumi", &lumi_);
  gunT_->Branch("event", &event_);
  gunT_->Branch("pdg", &gPdg_);
  gunT_->Branch("key", &gKey_);
  gunT_->Branch("pt", &gPt_);
  gunT_->Branch("eta", &gEta_);
  gunT_->Branch("phi", &gPhi_);
  gunT_->Branch("e", &gE_);
  gunT_->Branch("dep", &gDep_);
}

void EGStudyTruthProbe::beginRun(const edm::Run &, const edm::EventSetup &es) {
  tools_.setGeometry(es.getData(ticlGeomToken_), es.getData(ticlGeomLookupToken_), es.getData(ticlGeomLayersToken_));
}

EGStudyTruthProbe::Truth EGStudyTruthProbe::truthOf(const Hits &hits) const {
  Truth t;
  std::unordered_map<int, double> byObj;
  for (auto const &[id, f] : hits) {
    auto itE = recHitEnergy_.find(id);
    if (itE != recHitEnergy_.end())
      t.recoE += f * itE->second;
    auto it = cellTruth_.find(id);
    if (it == cellTruth_.end())
      continue;
    for (auto const &[q, e] : it->second) {
      auto itO = cbObj_.find(q);
      if (itO != cbObj_.end())
        byObj[itO->second] += f * e;
    }
  }
  auto const &parts = graph_->particles();
  double domDep = 0., sigDep = 0., puEM = 0., puHad = 0.;
  int sigKey = -1;
  for (auto const &[obj, e] : byObj) {
    t.dep += e;
    auto const &pd = parts[obj];
    const int sp = speciesOfPdg(pd.pdgId);
    if (sp == 0)
      t.emDep += e;
    else if (sp == 1)
      t.chDep += e;
    else if (sp == 2)
      t.nhDep += e;
    if (pd.isSignal()) {
      t.sigDep += e;
      if (e > sigDep) {
        sigDep = e;
        sigKey = obj;
      }
    } else if (sp == 0) {
      puEM = std::max(puEM, e);
    } else {
      puHad = std::max(puHad, e);
    }
    if (e > domDep) {
      domDep = e;
      t.domKey = obj;
    }
  }
  for (auto const &[obj, e] : byObj)
    if (t.dep > 0.f && e >= 0.1 * t.dep)
      ++t.nObj10;
  t.puEMLead = puEM;
  t.puHadLead = puHad;
  if (t.domKey >= 0) {
    auto const &pd = parts[t.domKey];
    t.domPdg = pd.pdgId;
    t.domSignal = pd.isSignal() ? 1 : 0;
    t.domE = pd.momentum.E();
    t.domEta = pd.momentum.E() > 0. ? pd.momentum.eta() : 0.f;
    t.domPhi = pd.momentum.E() > 0. ? pd.momentum.phi() : 0.f;
    t.domDep = domDep;
    auto itD = objDep_.find(t.domKey);
    t.domTot = itD == objDep_.end() ? 0.f : itD->second;
    const auto parents = truth::Particle(graph_, t.domKey).parents();
    t.domMotherPdg = parents.empty() ? 0 : parts[parents[0].id()].pdgId;
  }
  if (sigKey >= 0) {
    t.sigKey = sigKey;
    t.sigObjDep = sigDep;
    auto itS = objDep_.find(sigKey);
    t.sigObjTot = itS == objDep_.end() ? 0.f : itS->second;
  }
  return t;
}

void EGStudyTruthProbe::fillRow(
    int kind, int coll, int idx, float pt, float eta, float phi, float e, float rawE, const Truth &t) {
  kind_ = kind;
  coll_ = coll;
  idx_ = idx;
  pt_ = pt;
  eta_ = eta;
  phi_ = phi;
  e_ = e;
  rawE_ = rawE;
  t_ = t;
  objT_->Fill();
}

double EGStudyTruthProbe::trackIso(const std::vector<reco::Track> &tracks, float eta, float phi) const {
  double s = 0.;
  for (auto const &tk : tracks) {
    if (tk.pt() < 1.)
      continue;
    const double d2 = reco::deltaR2(tk.eta(), tk.phi(), eta, phi);
    if (d2 < 0.3 * 0.3 && d2 > 0.04 * 0.04)
      s += tk.pt();
  }
  return s;
}

void EGStudyTruthProbe::analyze(const edm::Event &event, const edm::EventSetup &) {
  auto const &lcs = event.get(lcToken_);
  auto const &graph = event.get(graphToken_);
  auto const &index = event.get(hitIndexToken_);
  auto const &tracks = event.get(tracksToken_);
  graph_ = &graph;
  run_ = event.id().run();
  lumi_ = event.id().luminosityBlock();
  event_ = event.id().event();
  const unsigned int nP = graph.particles().size();

  // Object of each particle: the nearest reconstructable-final-state ancestor (the particle itself first).
  objectCache_.assign(nP, -2);
  auto objectOf = [&](unsigned int p) -> int {
    if (objectCache_[p] != -2)
      return objectCache_[p];
    int found = -1;
    std::vector<unsigned int> walk{p};
    unsigned int cur = p;
    for (int depth = 0; depth < 64; ++depth) {
      if (graph.particles()[cur].isAtLevel(truth::LevelFlag::ReconstructableFinalState)) {
        found = static_cast<int>(cur);
        break;
      }
      const auto parents = truth::Particle(&graph, cur).parents();
      if (parents.empty())
        break;
      cur = parents[0].id();
      if (objectCache_[cur] >= 0) {
        found = objectCache_[cur];
        break;
      }
      walk.push_back(cur);
    }
    if (found < 0) {
      objectCache_[p] = static_cast<int>(p);
      return static_cast<int>(p);
    }
    for (auto w : walk)
      objectCache_[w] = found;
    return found;
  };

  cellTruth_.clear();
  recHitEnergy_.clear();
  cbObj_.clear();
  objDep_.clear();
  std::vector<truth::LogicalGraphHitIndex::Hit> hits;
  for (unsigned int p = 0; p < nP; ++p) {
    auto const &pd = graph.particles()[p];
    if (!pd.isAtLevel(truth::LevelFlag::CaloBoundary))
      continue;
    hits.clear();
    index.appendSubgraphHits(truth::HitChannel::Calo, p, hits);
    bool any = false;
    for (auto const &h : hits)
      if (isHGCal(h.detId) && h.energy > 0.f) {
        cellTruth_[h.detId].emplace_back(static_cast<int>(p), h.energy);
        any = true;
      }
    if (any)
      cbObj_[p] = objectOf(p);
  }
  for (auto const &token : recHitTokens_)
    for (auto const &rh : event.get(token))
      recHitEnergy_[rh.id().rawId()] = rh.energy();
  for (auto &[id, partsInCell] : cellTruth_) {
    auto itRh = recHitEnergy_.find(id);
    double simTot = 0.;
    for (auto const &[q, e] : partsInCell)
      simTot += e;
    for (auto &[q, e] : partsInCell) {
      e = itRh == recHitEnergy_.end() ? 0.f : static_cast<float>(itRh->second * e / simTot);
      objDep_[cbObj_[q]] += e;
    }
  }

  // Gun particles and their truth objects (match by pdg id and momentum, each graph particle used once).
  std::vector<char> usedKey(nP, 0);
  for (auto const &g : event.get(genToken_)) {
    if (g.status() != 1 || !g.isPromptFinalState() || (std::abs(g.pdgId()) != 11 && g.pdgId() != 22) ||
        std::abs(g.eta()) < 1.5 || std::abs(g.eta()) > 3.)
      continue;
    gKey_ = -1;
    for (unsigned int p = 0; p < nP; ++p) {
      auto const &pd = graph.particles()[p];
      if (usedKey[p] || !pd.isSignal() || pd.pdgId != g.pdgId())
        continue;
      if (std::abs(pd.momentum.E() - g.energy()) <= 1e-4 * std::max(1., g.energy()) &&
          reco::deltaR2(pd.momentum.eta(), pd.momentum.phi(), g.eta(), g.phi()) < 1e-6) {
        gKey_ = objectOf(p);
        usedKey[p] = 1;
        break;
      }
    }
    gPdg_ = g.pdgId();
    gPt_ = g.pt();
    gEta_ = g.eta();
    gPhi_ = g.phi();
    gE_ = g.energy();
    auto itD = gKey_ >= 0 ? objDep_.find(gKey_) : objDep_.end();
    gDep_ = itD == objDep_.end() ? 0.f : itD->second;
    gunT_->Fill();
  }

  const auto &tiles = event.get(tilesToken_);
  auto scHits = [](const reco::SuperCluster &sc) {
    Hits h;
    for (auto const &c : sc.clusters())
      for (auto const &[id, f] : c->hitsAndFractions())
        h.emplace_back(id.rawId(), f);
    return h;
  };
  auto tsHits = [&lcs](const ticl::Trackster &ts) {
    Hits h;
    for (std::size_t j = 0; j < ts.vertices().size(); ++j) {
      const auto v = ts.vertices()[j];
      if (v >= lcs.size())
        continue;
      const float m = ts.vertex_multiplicity().size() > j && ts.vertex_multiplicity()[j] > 0.f
                          ? 1.f / ts.vertex_multiplicity()[j]
                          : 1.f;
      for (auto const &[id, f] : lcs[v].hitsAndFractions())
        h.emplace_back(id.rawId(), f * m);
    }
    return h;
  };

  // Superclusters. x: 0 corrected energy, 1 clusters, 2 seed cluster energy / raw, 3 eta width, 4 phi width,
  // 5 track isolation, 6 HGCAL isolation (sum Et of layer clusters in dR 0.3 minus raw Et), 7 e/gamma trackster index,
  // 8-38 the score features 0-30 of EGammaSuperClusterFeatures.h (with the e/gamma trackster, else 0), 39 number of
  // tracks with pT > 2 GeV within dR 0.05, 40-43 the score features 31-34.
  namespace egsc = ticl::egammaSuperCluster;
  const auto &hitMap = event.get(hitMapToken_);
  const edm::MultiSpan<HGCRecHit> eventHits(event.get(hitsToken_));
  const std::vector<ticl::Trackster> *egTsForFeatures = nullptr;
  if (useEGammaTracksters_)
    egTsForFeatures = &event.get(egTsToken_);
  // Interpretation type of each e/gamma trackster (11: track in electron mode, else the neutral pdg id).
  std::unordered_map<int, int> typeOfFinal;
  const std::vector<int> *egIndexForFeatures = nullptr;
  if (useEGammaTracksters_) {
    egIndexForFeatures = &event.get(egIndexToken_);
    auto const &nIdx = event.get(neutralIdxToken_);
    auto const &nPdg = event.get(neutralPdgToken_);
    auto const &t2t = event.get(trackToTracksterToken_);
    auto const &tMode = event.get(trackModeToken_);
    for (std::size_t k = 0; k < nIdx.size(); ++k)
      typeOfFinal[nIdx[k]] = nPdg[k];
    for (std::size_t t = 0; t < tMode.size(); ++t)
      if (tMode[t] == 3 && t2t[t] >= 0)
        typeOfFinal[t2t[t]] = 11;
  }
  auto trackTiles = std::make_unique<egsc::TrackTiles>();
  egsc::fillTrackTiles(tracks, *trackTiles);
  std::vector<edm::ProductID> scIds;
  for (unsigned int c = 0; c < scTokens_.size(); ++c) {
    auto h = event.getHandle(scTokens_[c]);
    scIds.push_back(h.isValid() ? h.id() : edm::ProductID());
    if (!h.isValid())
      continue;
    auto hIdx = event.getHandle(scIndexTokens_[c]);
    for (unsigned int i = 0; i < h->size(); ++i) {
      auto const &sc = (*h)[i];
      const float eta = sc.position().eta(), phi = sc.position().phi();
      const float rawEt = sc.rawEnergy() / std::cosh(eta);
      if (std::abs(eta) < 1.5 || rawEt < minObjectPt_)
        continue;
      std::fill(x_, x_ + kNX, 0.f);
      x_[0] = sc.correctedEnergy();
      x_[1] = sc.clustersSize();
      x_[2] = sc.rawEnergy() > 0. ? sc.seed()->energy() / sc.rawEnergy() : 0.f;
      x_[3] = sc.etaWidth();
      x_[4] = sc.phiWidth();
      x_[5] = trackIso(tracks, eta, phi);
      x_[6] = isoSumEt(tiles, lcs, eta, phi, 0.3) - rawEt;
      x_[7] = hIdx.isValid() && i < hIdx->size() ? (*hIdx)[i] : -1;
      if (egTsForFeatures != nullptr && x_[7] >= 0 && x_[7] < egTsForFeatures->size()) {
        const auto &ts = (*egTsForFeatures)[x_[7]];
        const auto regression =
            egsc::regressionFeatures(sc, ts, egsc::cellFeatures(sc, hitMap, eventHits, tools_), eventHits.size());
        const auto isolation = egsc::isolationEt(sc, tiles, lcs, egsc::kIsolationInnerDR, egsc::kIsolationOuterDR);
        int type = 0;
        if (x_[7] < egIndexForFeatures->size()) {
          auto it = typeOfFinal.find((*egIndexForFeatures)[x_[7]]);
          type = it == typeOfFinal.end() ? 0 : it->second;
        }
        const auto f = egsc::scoreFeatures(
            regression, sc, ts, isolation, type == 11, type == 111, egsc::trackFeatures(sc, tracks, *trackTiles));
        for (std::size_t j = 0; j < 31; ++j)
          x_[8 + j] = f[j];
        for (std::size_t j = 31; j < f.size(); ++j)
          x_[40 + j - 31] = f[j];
      }
      int nNearTk = 0;
      for (auto const &tk : tracks)
        if (tk.pt() > 2. && reco::deltaR2(tk.eta(), tk.phi(), eta, phi) < 0.05 * 0.05)
          ++nNearTk;
      x_[39] = nNearTk;
      fillRow(0, c, i, sc.energy() / std::cosh(eta), eta, phi, sc.energy(), sc.rawEnergy(), truthOf(scHits(sc)));
    }
  }
  auto scColl = [&scIds](edm::ProductID id) -> int {
    auto it = std::find(scIds.begin(), scIds.end(), id);
    return it == scIds.end() ? -1 : static_cast<int>(it - scIds.begin());
  };

  // Photons. x: 0 supercluster collection, 1 supercluster key, 2 H/E, 3 track isolation, 4 HGCAL isolation, 5 r9,
  // 6 hollow-cone track isolation of the photon, 7 sigma ieta ieta, 8 SC raw energy.
  if (auto h = event.getHandle(photonToken_); h.isValid()) {
    for (unsigned int i = 0; i < h->size(); ++i) {
      auto const &p = (*h)[i];
      if (p.pt() < minObjectPt_)
        continue;
      std::fill(x_, x_ + kNX, 0.f);
      auto sc = p.superCluster();
      x_[0] = scColl(sc.id());
      x_[1] = sc.key();
      x_[2] = p.hadronicOverEm();
      x_[3] = trackIso(tracks, p.eta(), p.phi());
      x_[4] = isoSumEt(tiles, lcs, sc->position().eta(), sc->position().phi(), 0.3) -
              sc->rawEnergy() / std::cosh(sc->position().eta());
      x_[5] = p.r9();
      x_[6] = p.trkSumPtHollowConeDR03();
      x_[7] = p.sigmaIetaIeta();
      x_[8] = sc->rawEnergy();
      fillRow(1, 0, i, p.pt(), p.eta(), p.phi(), p.energy(), sc->rawEnergy(), truthOf(scHits(*sc)));
    }
  }

  // Electrons. x: 0 supercluster collection, 1 supercluster key, 2 charge, 3 E_SC/p, 4 dEtaIn, 5 dPhiIn, 6 H/E,
  // 7 fbrem, 8 GSF missing inner hits, 9 dr03 track isolation, 10 track isolation, 11 HGCAL isolation, 12 ECAL-driven,
  // 13 tracker-driven, 14 GSF pT mode, 15 GSF normalized chi2, 16 SC raw energy, 17 GSF valid hits.
  if (auto h = event.getHandle(electronToken_); h.isValid()) {
    for (unsigned int i = 0; i < h->size(); ++i) {
      auto const &el = (*h)[i];
      if (el.pt() < minObjectPt_)
        continue;
      std::fill(x_, x_ + kNX, 0.f);
      auto sc = el.superCluster();
      x_[0] = scColl(sc.id());
      x_[1] = sc.key();
      x_[2] = el.charge();
      x_[3] = el.eSuperClusterOverP();
      x_[4] = el.deltaEtaSuperClusterTrackAtVtx();
      x_[5] = el.deltaPhiSuperClusterTrackAtVtx();
      x_[6] = el.hadronicOverEm();
      x_[7] = el.fbrem();
      auto gsf = el.gsfTrack();
      x_[8] = gsf->hitPattern().numberOfLostHits(reco::HitPattern::MISSING_INNER_HITS);
      x_[9] = el.dr03TkSumPt();
      x_[10] = trackIso(tracks, el.eta(), el.phi());
      x_[11] = isoSumEt(tiles, lcs, sc->position().eta(), sc->position().phi(), 0.3) -
               sc->rawEnergy() / std::cosh(sc->position().eta());
      x_[12] = el.ecalDrivenSeed();
      x_[13] = el.trackerDrivenSeed();
      x_[14] = gsf->ptMode();
      x_[15] = gsf->normalizedChi2();
      x_[16] = sc->rawEnergy();
      x_[17] = gsf->numberOfValidHits();
      fillRow(2, 0, i, el.pt(), el.eta(), el.phi(), el.energy(), sc->rawEnergy(), truthOf(scHits(*sc)));
    }
  }

  // TICL candidates in the endcaps. x: 0 pdg id, 1 charge, 2 tracksters, 3 has a track, 4 raw energy of the
  // tracksters, 5 time, 6 time error, 7 P(photon), 8 P(electron), 9 track isolation, 10 HGCAL isolation,
  // 11 energy of the candidate of the alternative collection on the same trackster (-1: none), 12 trackster index.
  if (auto h = event.getHandle(candToken_); h.isValid()) {
    std::unordered_map<unsigned int, float> altEnergy;
    if (useCandAlt_)
      for (auto const &c : event.get(candAltToken_))
        if (c.charge() == 0 && !c.tracksters().empty())
          altEnergy[c.tracksters()[0].key()] = c.energy();
    for (unsigned int i = 0; i < h->size(); ++i) {
      auto const &c = (*h)[i];
      if (c.pt() < minObjectPt_ || std::abs(c.eta()) < 1.5 || std::abs(c.eta()) > 3.)
        continue;
      std::fill(x_, x_ + kNX, 0.f);
      Hits hh;
      double raw = 0.;
      for (auto const &ts : c.tracksters()) {
        auto th = tsHits(*ts);
        hh.insert(hh.end(), th.begin(), th.end());
        raw += ts->raw_energy();
      }
      x_[0] = c.pdgId();
      x_[1] = c.charge();
      x_[2] = c.tracksters().size();
      x_[3] = c.trackPtr().isNonnull();
      x_[4] = raw;
      x_[5] = c.time();
      x_[6] = c.timeError();
      x_[7] = c.id_probability(ticl::Trackster::ParticleType::photon);
      x_[8] = c.id_probability(ticl::Trackster::ParticleType::electron);
      x_[9] = trackIso(tracks, c.eta(), c.phi());
      x_[10] = isoSumEt(tiles, lcs, c.eta(), c.phi(), 0.3) - raw / std::cosh(c.eta());
      x_[11] = -1.f;
      x_[12] = c.tracksters().empty() ? -1 : c.tracksters()[0].key();
      if (useCandAlt_ && !c.tracksters().empty()) {
        auto it = altEnergy.find(c.tracksters()[0].key());
        if (it != altEnergy.end())
          x_[11] = it->second;
      }
      fillRow(3, 0, i, c.pt(), c.eta(), c.phi(), c.energy(), c.rawEnergy(), truthOf(hh));
    }
  }

  // E/gamma tracksters. x: 0 type (11 electron track, else the neutral pdg id), 1 layer clusters, 2 raw EM energy,
  // 3 time, 4 time error, 5-12 PID probabilities, 13-15 PCA sigmas, 16 row of the supercluster of collection 0 (-1),
  // 17 track isolation, 18 HGCAL isolation, 19 regressed energy, 20 final trackster index.
  if (useEGammaTracksters_) {
    auto const &egTs = event.get(egTsToken_);
    auto const &egIdx = event.get(egIndexToken_);
    auto const &trackToTrackster = event.get(trackToTracksterToken_);
    auto const &trackMode = event.get(trackModeToken_);
    auto const &neutralIdx = event.get(neutralIdxToken_);
    auto const &neutralPdg = event.get(neutralPdgToken_);
    std::unordered_map<int, int> typeOf;
    for (std::size_t k = 0; k < neutralIdx.size(); ++k)
      typeOf[neutralIdx[k]] = neutralPdg[k];
    for (std::size_t t = 0; t < trackMode.size(); ++t)
      if (trackMode[t] == 3 && trackToTrackster[t] >= 0)
        typeOf[trackToTrackster[t]] = 11;
    std::unordered_map<int, int> scRowOf;
    if (!scTokens_.empty())
      if (auto hIdx = event.getHandle(scIndexTokens_[0]); hIdx.isValid())
        for (unsigned int r = 0; r < hIdx->size(); ++r)
          scRowOf[(*hIdx)[r]] = r;
    for (unsigned int k = 0; k < egTs.size(); ++k) {
      auto const &ts = egTs[k];
      const float eta = ts.barycenter().eta(), phi = ts.barycenter().phi();
      const float pt = ts.raw_energy() / std::cosh(eta);
      if (pt < minObjectPt_)
        continue;
      std::fill(x_, x_ + kNX, 0.f);
      const int fin = k < egIdx.size() ? egIdx[k] : -1;
      auto itT = typeOf.find(fin);
      x_[0] = itT == typeOf.end() ? 0 : itT->second;
      x_[1] = ts.vertices().size();
      x_[2] = ts.raw_em_energy();
      x_[3] = ts.time();
      x_[4] = ts.timeError();
      for (int j = 0; j < 8; ++j)
        x_[5 + j] = ts.id_probabilities(j);
      x_[13] = ts.sigmasPCA()[0];
      x_[14] = ts.sigmasPCA()[1];
      x_[15] = ts.sigmasPCA()[2];
      auto itS = scRowOf.find(k);
      x_[16] = itS == scRowOf.end() ? -1 : itS->second;
      x_[17] = trackIso(tracks, eta, phi);
      x_[18] = isoSumEt(tiles, lcs, eta, phi, 0.3) - pt;
      x_[19] = ts.regressed_energy();
      x_[20] = fin;
      fillRow(4, 0, k, pt, eta, phi, ts.raw_energy(), ts.raw_energy(), truthOf(tsHits(ts)));
    }
  }
}

DEFINE_FWK_MODULE(EGStudyTruthProbe);
