// Study dumper: one tree row per event with the generator photons and, for each TICL e/gamma supercluster within
// dR 0.3 of a generator photon, its energies and the regression features of EGammaSuperClusterFeatures.h, computed
// with the same functions as TICLEGammaSuperClustersProducer.

#include <cmath>
#include <cstddef>
#include <unordered_map>
#include <vector>

#include "CommonTools/UtilAlgos/interface/TFileService.h"
#include "DataFormats/Common/interface/MultiSpan.h"
#include "DataFormats/Common/interface/RefProdVector.h"
#include "DataFormats/EgammaReco/interface/SuperCluster.h"
#include "DataFormats/EgammaReco/interface/SuperClusterFwd.h"
#include "DataFormats/HGCRecHit/interface/HGCRecHitCollections.h"
#include "DataFormats/HGCalReco/interface/Trackster.h"
#include "DataFormats/HepMCCandidate/interface/GenParticle.h"
#include "DataFormats/Math/interface/deltaR.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/Run.h"
#include "FWCore/Framework/interface/one/EDAnalyzer.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "Geometry/Records/interface/CaloGeometryRecord.h"
#include "RecoLocalCalo/HGCalRecAlgos/interface/TICLGeomTools.h"
#include "RecoTICL/ParticleFlow/interface/EGammaSuperClusterFeatures.h"
#include "TTree.h"

class EGammaSCFeatureDumper : public edm::one::EDAnalyzer<edm::one::SharedResources, edm::one::WatchRuns> {
public:
  explicit EGammaSCFeatureDumper(const edm::ParameterSet &ps);
  static void fillDescriptions(edm::ConfigurationDescriptions &descriptions);

private:
  void beginRun(const edm::Run &, const edm::EventSetup &es) override;
  void endRun(const edm::Run &, const edm::EventSetup &) override {}
  void analyze(const edm::Event &evt, const edm::EventSetup &) override;

  const edm::EDGetTokenT<reco::SuperClusterCollection> sc_token_;
  const edm::EDGetTokenT<std::vector<int>> index_token_;
  const edm::EDGetTokenT<std::vector<ticl::Trackster>> ts_token_;
  const edm::EDGetTokenT<std::vector<reco::GenParticle>> gen_token_;
  const edm::EDGetTokenT<std::unordered_map<DetId, const unsigned int>> hitMap_token_;
  const edm::EDGetTokenT<edm::RefProdVector<HGCRecHitCollection>> hits_token_;
  const edm::ESGetToken<TICLGeomHost, CaloGeometryRecord> ticlGeomToken_;
  const edm::ESGetToken<TICLGeomLookupHost, CaloGeometryRecord> ticlGeomLookupToken_;
  const edm::ESGetToken<TICLGeomLayersHost, CaloGeometryRecord> ticlGeomLayersToken_;
  ticlgeom::Tools tools_;

  TTree *tree_ = nullptr;
  unsigned int run_ = 0, lumi_ = 0;
  unsigned long long event_ = 0;
  unsigned int nHits_ = 0;
  std::vector<float> genE_, genEta_, genPhi_, genVz_;
  std::vector<float> scRaw_, scE_, scEta_, scPhi_, scX_, scY_, scZ_, scFeatures_;
};

EGammaSCFeatureDumper::EGammaSCFeatureDumper(const edm::ParameterSet &ps)
    : sc_token_(consumes(ps.getParameter<edm::InputTag>("superClusters"))),
      index_token_(consumes(edm::InputTag(ps.getParameter<edm::InputTag>("superClusters").label(), "tracksterIndex"))),
      ts_token_(consumes(ps.getParameter<edm::InputTag>("eGammaTracksters"))),
      gen_token_(consumes(ps.getParameter<edm::InputTag>("genParticles"))),
      hitMap_token_(consumes(ps.getParameter<edm::InputTag>("hitMap"))),
      hits_token_(consumes(ps.getParameter<edm::InputTag>("hits"))),
      ticlGeomToken_(esConsumes<TICLGeomHost, CaloGeometryRecord, edm::Transition::BeginRun>()),
      ticlGeomLookupToken_(esConsumes<TICLGeomLookupHost, CaloGeometryRecord, edm::Transition::BeginRun>()),
      ticlGeomLayersToken_(esConsumes<TICLGeomLayersHost, CaloGeometryRecord, edm::Transition::BeginRun>()) {
  usesResource(TFileService::kSharedResource);
  edm::Service<TFileService> fs;
  tree_ = fs->make<TTree>("sc", "sc");
  tree_->Branch("run", &run_);
  tree_->Branch("lumi", &lumi_);
  tree_->Branch("event", &event_);
  tree_->Branch("nHits", &nHits_);
  tree_->Branch("genE", &genE_);
  tree_->Branch("genEta", &genEta_);
  tree_->Branch("genPhi", &genPhi_);
  tree_->Branch("genVz", &genVz_);
  tree_->Branch("scRaw", &scRaw_);
  tree_->Branch("scE", &scE_);
  tree_->Branch("scEta", &scEta_);
  tree_->Branch("scPhi", &scPhi_);
  tree_->Branch("scX", &scX_);
  tree_->Branch("scY", &scY_);
  tree_->Branch("scZ", &scZ_);
  tree_->Branch("scFeatures", &scFeatures_);
}

void EGammaSCFeatureDumper::beginRun(const edm::Run &, const edm::EventSetup &es) {
  tools_.setGeometry(es.getData(ticlGeomToken_), es.getData(ticlGeomLookupToken_), es.getData(ticlGeomLayersToken_));
}

void EGammaSCFeatureDumper::analyze(const edm::Event &evt, const edm::EventSetup &) {
  namespace egsc = ticl::egammaSuperCluster;
  run_ = evt.id().run();
  lumi_ = evt.id().luminosityBlock();
  event_ = evt.id().event();
  for (auto *v : {&genE_, &genEta_, &genPhi_, &genVz_, &scRaw_, &scE_, &scEta_, &scPhi_, &scX_, &scY_, &scZ_, &scFeatures_})
    v->clear();
  for (auto const &g : evt.get(gen_token_)) {
    if (g.status() != 1 || g.pdgId() != 22 || g.energy() < 5. || std::abs(g.eta()) < 1.5 || std::abs(g.eta()) > 3.)
      continue;
    genE_.push_back(g.energy());
    genEta_.push_back(g.eta());
    genPhi_.push_back(g.phi());
    genVz_.push_back(g.vz());
  }
  const auto &scs = evt.get(sc_token_);
  const auto &index = evt.get(index_token_);
  const auto &tss = evt.get(ts_token_);
  const auto &hitMap = evt.get(hitMap_token_);
  const edm::MultiSpan<HGCRecHit> hits(evt.get(hits_token_));
  nHits_ = hits.size();
  for (std::size_t i = 0; i < scs.size(); ++i) {
    const auto &sc = scs[i];
    bool near = false;
    for (std::size_t j = 0; j < genE_.size(); ++j)
      near = near || reco::deltaR(sc.eta(), sc.phi(), genEta_[j], genPhi_[j]) < 0.3;
    if (!near)
      continue;
    const auto cell = egsc::cellFeatures(sc, hitMap, hits, tools_);
    const auto features = egsc::regressionFeatures(sc, tss.at(index.at(i)), cell, hits.size());
    scRaw_.push_back(sc.rawEnergy());
    scE_.push_back(sc.energy());
    scEta_.push_back(sc.eta());
    scPhi_.push_back(sc.phi());
    scX_.push_back(sc.position().x());
    scY_.push_back(sc.position().y());
    scZ_.push_back(sc.position().z());
    scFeatures_.insert(scFeatures_.end(), features.begin(), features.end());
  }
  tree_->Fill();
}

void EGammaSCFeatureDumper::fillDescriptions(edm::ConfigurationDescriptions &descriptions) {
  edm::ParameterSetDescription desc;
  desc.add<edm::InputTag>("superClusters", edm::InputTag("ticlEGammaSuperClusters"));
  desc.add<edm::InputTag>("eGammaTracksters", edm::InputTag("ticlEGammaTracksters"));
  desc.add<edm::InputTag>("genParticles", edm::InputTag("genParticles"));
  desc.add<edm::InputTag>("hitMap", edm::InputTag("recHitMapProducer", "hgcalRecHitMap"));
  desc.add<edm::InputTag>("hits", edm::InputTag("recHitMapProducer", "RefProdVectorHGCRecHitCollection"));
  descriptions.add("egammaSCFeatureDumper", desc);
}

DEFINE_FWK_MODULE(EGammaSCFeatureDumper);
