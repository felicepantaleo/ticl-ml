# Flat table of the final tracksters of the interpretation stage (tree fin) from an EDM file that holds them (a leg
# rev.root): the trackster-level shower shapes and the id probabilities. A C++ loop (cling) reads the EDM products.
# usage: extract_fin.py <EDM file> <output root file>
import sys
import ROOT

ROOT.gSystem.Load("libFWCoreFWLite")
ROOT.FWLiteEnabler.enable()
ROOT.gSystem.Load("libDataFormatsHGCalReco")
ROOT.gInterpreter.Declare(r'''
#include "DataFormats/Common/interface/Wrapper.h"
#include "DataFormats/HGCalReco/interface/Trackster.h"
#include "DataFormats/Provenance/interface/EventAuxiliary.h"
#include "TFile.h"
#include "TTree.h"
#include <cmath>
#include <string>
#include <vector>
namespace pidstudy {
using W = edm::Wrapper<std::vector<ticl::Trackster>>;
struct Row {
  int event, idx, nLC; float rawE, rawEmE, eta, absZ, time, timeErr, eig[3], sig[3], sigPCA[3], ax[3];
};
void branches(TTree* t, Row& r) {
  t->Branch("event", &r.event); t->Branch("idx", &r.idx); t->Branch("nLC", &r.nLC);
  t->Branch("rawE", &r.rawE); t->Branch("rawEmE", &r.rawEmE); t->Branch("eta", &r.eta); t->Branch("absZ", &r.absZ);
  t->Branch("time", &r.time); t->Branch("timeErr", &r.timeErr);
  t->Branch("eig", r.eig, "eig[3]/F"); t->Branch("sig", r.sig, "sig[3]/F"); t->Branch("sigPCA", r.sigPCA, "sigPCA[3]/F");
  t->Branch("axis", r.ax, "axis[3]/F");
}
void fill(Row& r, const ticl::Trackster& t, int ev, int i) {
  r.event = ev; r.idx = i; r.nLC = t.vertices().size(); r.rawE = t.raw_energy(); r.rawEmE = t.raw_em_energy();
  r.eta = t.barycenter().eta(); r.absZ = std::abs(t.barycenter().z()); r.time = t.time(); r.timeErr = t.timeError();
  for (int k = 0; k < 3; ++k) { r.eig[k] = t.eigenvalues()[k]; r.sig[k] = t.sigmas()[k]; r.sigPCA[k] = t.sigmasPCA()[k]; }
  auto a = t.eigenvectors(0); r.ax[0] = a.x(); r.ax[1] = a.y(); r.ax[2] = a.z();
}
int run(const std::string& in, const std::string& out) {
  TFile f(in.c_str()); TTree* ev = (TTree*)f.Get("Events");
  W* wf = nullptr; ev->SetBranchAddress("ticlTracksters_ticlTracksterInterpretations__RECO.", &wf);
  edm::EventAuxiliary* aux = nullptr; ev->SetBranchAddress("EventAuxiliary", &aux);
  TFile fo(out.c_str(), "RECREATE");
  TTree hi("hi", "CLUE3DHigh tracksters"), fin("fin", "final tracksters");
  Row r; branches(&hi, r); branches(&fin, r);
  float p[8]; hi.Branch("pid", p, "pid[8]/F"); fin.Branch("pid", p, "pid[8]/F");
  for (Long64_t e = 0; e < ev->GetEntries(); ++e) {
    ev->GetEntry(e);
    const int id = aux->event();
    const auto& vf = *wf->product();
    for (size_t i = 0; i < vf.size(); ++i) {
      fill(r, vf[i], id, i);
      for (int c = 0; c < 8; ++c) p[c] = vf[i].id_probabilities(c);
      fin.Fill();
    }
  }
  fo.Write();
  return 0;
}
}
''')
ROOT.pidstudy.run(sys.argv[1], sys.argv[2])
print(f"{sys.argv[2]} written")
