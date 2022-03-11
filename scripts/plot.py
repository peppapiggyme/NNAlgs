
#region imports
import sys, os

print("importing uproot")
import uproot

print("importing numpy")
import numpy as np 

print("importing pandas")
import pandas as pd

# print("importing seaborn")
# import seaborn as sns

print("importing matplotlib")
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator

print("importing scipy")
from scipy import stats
from math import sqrt
#endregion


sel_vars = [
    "TauJets.mcEventNumber",
    "TauJets.IsTruthMatched", 
    "TauJets.truthDecayMode"
]

data = {
    "TauTrack": [
        "TauTrack.dphiECal",
        "TauTrack.dphi",
        "TauTrack.detaECal",
        "TauTrack.deta",
        "TauTrack.pt",
        "TauTrack.jetpt", 
        "TauTrack.d0TJVA",
        "TauTrack.d0SigTJVA",
        "TauTrack.z0sinthetaTJVA",
        "TauTrack.z0sinthetaSigTJVA", 
        #"TauTrack.rnn_chargedScore",
        #"TauTrack.rnn_isolationScore",
        #"TauTrack.rnn_conversionScore",
        #"TauTrack.rnn_fakeScore", 
    ],
    "NeutralPFO": [
        "NeutralPFO.dphiECal",
        "NeutralPFO.dphi",
        "NeutralPFO.detaECal",
        "NeutralPFO.deta",
        "NeutralPFO.pt",
        "NeutralPFO.jetpt",
        "NeutralPFO.FIRST_ETA",
        "NeutralPFO.SECOND_R",
        "NeutralPFO.DELTA_THETA",
        "NeutralPFO.CENTER_LAMBDA",
        "NeutralPFO.LONGITUDINAL",
        "NeutralPFO.SECOND_ENG_DENS",
        "NeutralPFO.ENG_FRAC_CORE",
        "NeutralPFO.NPosECells_EM1",
        "NeutralPFO.NPosECells_EM2",
        "NeutralPFO.energy_EM1",
        "NeutralPFO.energy_EM2",
        "NeutralPFO.EM1CoreFrac",
        "NeutralPFO.firstEtaWRTClusterPosition_EM1",
        "NeutralPFO.firstEtaWRTClusterPosition_EM2",
        "NeutralPFO.secondEtaWRTClusterPosition_EM1",
        "NeutralPFO.secondEtaWRTClusterPosition_EM2",
    ],
    "ShotPFO": [
        "ShotPFO.dphiECal",
        "ShotPFO.dphi",
        "ShotPFO.detaECal",
        "ShotPFO.deta",
        "ShotPFO.pt",
        "ShotPFO.jetpt", 
    ],
    "ConvTrack": [
        "ConvTrack.dphiECal",
        "ConvTrack.dphi",
        "ConvTrack.detaECal",
        "ConvTrack.deta",
        "ConvTrack.pt",
        "ConvTrack.jetpt", 
        "ConvTrack.d0TJVA",
        "ConvTrack.d0SigTJVA",
        "ConvTrack.z0sinthetaTJVA",
        "ConvTrack.z0sinthetaSigTJVA", 
        #"ConvTrack.rnn_chargedScore",
        #"ConvTrack.rnn_isolationScore",
        #"ConvTrack.rnn_conversionScore",
        #"ConvTrack.rnn_fakeScore", 
    ],
    "Label": [
        "TauJets.truthDecayMode"
    ],
    "Weight": [
        "TauJets.beamSpotWeight"
    ],
}

N_ENTRIES = 2000000
# N_ENTRIES = 10000

n_steps = {
    "ChargedPFO": 3,
    "NeutralPFO": 10,
    "ShotPFO": 6,
    "ConvTrack": 4,
    "Label": None,
}

def trans(arr, radius=0.4):
    # clamp values that are more than 0.4, never happen
    arr[arr>0.4] = 0.399
    arr[arr<-0.4] = -0.399
    # transform so that the centre is zoomed in
    arr[arr>0] = np.sqrt(radius**2 - (arr[arr>0]-radius)**2)
    arr[arr<0] = -1 * np.sqrt(radius**2 - (arr[arr<0]+radius)**2)

    return arr

def getline_1(radius):
    N = 100
    radius /= 0.4
    t = np.linspace(0, 0.5 * np.pi, N)
    x, y = 0.4 * np.sqrt(1-np.square(1-radius * np.cos(t))), 0.4 * np.sqrt(1-np.square(1-radius * np.sin(t)))
    
    return x, y

def getline_2(radius):
    N = 400
    radius /= 0.4
    t = np.linspace(0.5 * np.pi, np.pi, N)
    x, y = -0.4 * np.sqrt(1-np.square(1+radius * np.cos(t))), 0.4 * np.sqrt(1-np.square(1-radius * np.sin(t)))
    
    return x, y

def getline_3(radius):
    N = 400
    radius /= 0.4
    t = np.linspace(np.pi, 1.5 * np.pi, N)
    x, y = -0.4 * np.sqrt(np.abs(1-np.square(1+radius * np.cos(t)))), -0.4 * np.sqrt(np.abs(1-np.square(1+radius * np.sin(t))))
    
    return x, y

def getline_4(radius):
    N = 400
    radius /= 0.4
    t = np.linspace(1.5 * np.pi, 2 * np.pi, N)
    x, y = 0.4 * np.sqrt(np.abs(1-np.square(1-radius * np.cos(t)))), -0.4 * np.sqrt(np.abs(1-np.square(1+radius * np.sin(t))))
    
    return x, y

def getline(radius):
    x1, y1 = getline_1(radius)
    x2, y2 = getline_2(radius)
    x3, y3 = getline_3(radius)
    x4, y4 = getline_4(radius)

    return np.concatenate((x1, x2, x3, x4)), np.concatenate((y1, y2, y3, y4))

def get_removed_indices_train(df):

    return df[(df["TauJets.mcEventNumber"] % 5 == 0) 
        | (df["TauJets.mcEventNumber"] % 5 == 1)].index

def get_removed_indices_decaymode(df, idx=-1):

    return df[(df["TauJets.mcEventNumber"] % 5 == 0) 
        | (df["TauJets.mcEventNumber"] % 5 == 1)
        | (df["TauJets.truthDecayMode"] != idx)].index

def cleaned_arr_simple(var, removed_indices, n_entries=N_ENTRIES):
    df = tree.pandas.df([var], entrystop=n_entries, flatten=False)
    df.drop(removed_indices, inplace=True)
    df.reset_index(drop=True, inplace=True)
    arr = np.asarray(df, dtype=np.float32)

    return arr

def cleaned_arr(var, removed_indices, n_entries=N_ENTRIES):
    df = tree.pandas.df([var], entrystop=n_entries, flatten=False)
    df.drop(removed_indices, inplace=True)
    df.reset_index(drop=True, inplace=True)
    longest = max([len(a) for a in df[var]])
    arr = np.zeros((1, len(df), longest), np.float32)
    list_here = list()
    for n, row in df.iterrows():
        list_here.append(np.pad(row[var], (0, longest - len(row[var])), 'constant'))
    arr[0, :, :] = np.vstack(list_here)  # (*, B, T)
    arr = arr.transpose((1, 2, 0))  # (B, T, *)

    return arr

def get_x_y_r(map_p4, name, i_event):

    i_eta = map_p4["{}.deta".format(name)][i_event]
    i_phi = map_p4["{}.dphi".format(name)][i_event]
    i_ratio = map_p4["{}.ratio".format(name)][i_event]

    return np.concatenate((i_eta, i_phi, i_ratio), axis=1)


tree = uproot.open("/publicfs/atlas/atlasnew/higgs/hh2X/zhangbw/MxAOD/r22-03/Gtt_ntuple.root")["tree"]
# print(tree)

df_sel = tree.pandas.df(sel_vars, entrystop=N_ENTRIES)
# print(df_sel)

sel_basic = get_removed_indices_train(df_sel)
sel_1p0n = get_removed_indices_decaymode(df_sel, 0)
sel_1p1n = get_removed_indices_decaymode(df_sel, 1)
sel_1pXn = get_removed_indices_decaymode(df_sel, 2)
sel_3p0n = get_removed_indices_decaymode(df_sel, 3)
sel_3pXn = get_removed_indices_decaymode(df_sel, 4)


def draw_event_display(sel, i_event, output_name, extra_label, extrapolated=False):

    dphi = "dphiECal" if extrapolated else "dphi"
    deta = "detaECal" if extrapolated else "deta"

    x__p4 = {
        "TauTrack.dphi"    : trans(cleaned_arr("TauTrack.{}".format(dphi), sel, N_ENTRIES)),
        "TauTrack.deta"    : trans(cleaned_arr("TauTrack.{}".format(deta), sel, N_ENTRIES)),
        "TauTrack.ratio"   : cleaned_arr("TauTrack.pt", sel, N_ENTRIES) / cleaned_arr_simple("TauJets.truthPtVis", sel)[:,:,np.newaxis],
        "NeutralPFO.dphi"  : trans(cleaned_arr("NeutralPFO.{}".format(dphi), sel, N_ENTRIES)),
        "NeutralPFO.deta"  : trans(cleaned_arr("NeutralPFO.{}".format(deta), sel, N_ENTRIES)),
        "NeutralPFO.ratio" : cleaned_arr("NeutralPFO.pt", sel, N_ENTRIES) / cleaned_arr_simple("TauJets.truthPtVis", sel)[:,:,np.newaxis],
        "ConvTrack.dphi"   : trans(cleaned_arr("ConvTrack.{}".format(dphi), sel, N_ENTRIES)),
        "ConvTrack.deta"   : trans(cleaned_arr("ConvTrack.{}".format(deta), sel, N_ENTRIES)),
        "ConvTrack.ratio"  : cleaned_arr("ConvTrack.pt", sel, N_ENTRIES) / cleaned_arr_simple("TauJets.truthPtVis", sel)[:,:,np.newaxis],
        "ShotPFO.dphi"     : trans(cleaned_arr("ShotPFO.{}".format(dphi), sel, N_ENTRIES)),
        "ShotPFO.deta"     : trans(cleaned_arr("ShotPFO.{}".format(deta), sel, N_ENTRIES)),
        "ShotPFO.ratio"    : cleaned_arr("ShotPFO.pt", sel, N_ENTRIES) / cleaned_arr_simple("TauJets.truthPtVis", sel)[:,:,np.newaxis],
    }

    # print(x__p4["TauTrack.ratio"][i_event])
    # print(x__p4["NeutralPFO.ratio"][i_event])

    scatters = {
        (r"$\tau_{had}$ Tracks", "blue") : get_x_y_r(x__p4, "TauTrack", i_event),
        (r"Conversion Tracks", "purple") : get_x_y_r(x__p4, "ConvTrack", i_event),
        (r"Neutral PFOs", "orange") : get_x_y_r(x__p4, "NeutralPFO", i_event),
        (r"Shot PFOs", "green") : get_x_y_r(x__p4, "ShotPFO", i_event),
    }

    # print(scatters[(r"$\tau_{had}$ Tracks", "blue")])
    # print(scatters[(r"$\tau_{had}$ Tracks", "blue")][:,0])
    # print(scatters[(r"$\tau_{had}$ Tracks", "blue")][:,1])
    # print(scatters[(r"$\tau_{had}$ Tracks", "blue")][:,2])

    fig, ax = plt.subplots(figsize=(6, 6))

    plt.subplots_adjust(left=0.15, bottom=0.15, right=0.90, top=0.90)
    list_sca = []

    ax.plot(*getline(0.4), color='gray', linestyle='--', linewidth=1)
    ax.plot(*getline(0.2), color='gray', linestyle='--', linewidth=1)
    ax.plot(*getline(0.1), color='gray', linestyle='--', linewidth=1)

    plt.axvline(0, color='gray', linestyle='--', linewidth=1, zorder=1)
    plt.axhline(0, color='gray', linestyle='--', linewidth=1, zorder=2)

    for (label, color), array in scatters.items():
        sca = ax.scatter(array[:,0], array[:,1], c=color, s=np.sqrt(array[:,2])*5e3, label=label, alpha=0.5, zorder=10)
        sca = ax.scatter(array[:,0], array[:,1], c=color, s=6., label=label, alpha=0.5, zorder=11)
        list_sca.append(sca)

    plt.text(-0.46, 0.43, extra_label, color='black', fontsize=15)
    plt.text(0.01, -0.30, r"$\Delta R=0.10$", color='gray', fontsize=13)
    plt.text(0.01, -0.382, r"$\Delta R=0.20$", color='gray', fontsize=13)
    plt.text(0.01, -0.435, r"$\Delta R=0.40$", color='gray', fontsize=13)

    leg = ax.legend(
        handles=list_sca, 
        bbox_to_anchor=(0,1.02,1,0.2), 
        loc="lower left", 
        mode="expand", 
        borderaxespad=0, 
        ncol=2, fontsize=13
    )

    for handle in leg.legendHandles:
        handle.set_sizes([36.0])

    ax.add_artist(leg)

    ax.grid(False)
    plt.xlim((-0.48, 0.48))
    plt.ylim((-0.48, 0.48))

    referee = r"trackECal" if extrapolated else r"\tau_{had}"

    plt.xlabel(r"$\Delta\eta^{\prime}(object, " + referee + r")$", fontsize=20)
    plt.ylabel(r"$\Delta\phi^{\prime}(object, " + referee + r")$", fontsize=20)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)

    plt.savefig(output_name, bbox_inches='tight')
    plt.close()


def draw_input_variables(varname, varname_tex, nbins, xrange, output_name, log=False):

    print("plotting {}".format(varname))

    # draw the cleaned and flattened variables
    x__var = {
        r"$h^{\pm}$" : cleaned_arr(varname, sel_1p0n, N_ENTRIES),
        r"$h^{\pm}\pi^{0}$" : cleaned_arr(varname, sel_1p1n, N_ENTRIES),
        r"$h^{\pm}\geq2\pi^{0}$" : cleaned_arr(varname, sel_1pXn, N_ENTRIES),
        r"$3h^{\pm}$" : cleaned_arr(varname, sel_3p0n, N_ENTRIES),
        r"$3h^{\pm}\geq1\pi^{0}$" : cleaned_arr(varname, sel_3pXn, N_ENTRIES),
    }

    # mc_weight is 1., only this weight matters
    x__weight = {
        r"$h^{\pm}$" : cleaned_arr_simple("TauJets.beamSpotWeight", sel_1p0n, N_ENTRIES).flatten(),
        r"$h^{\pm}\pi^{0}$" : cleaned_arr_simple("TauJets.beamSpotWeight", sel_1p1n, N_ENTRIES).flatten(),
        r"$h^{\pm}\geq2\pi^{0}$" : cleaned_arr_simple("TauJets.beamSpotWeight", sel_1pXn, N_ENTRIES).flatten(),
        r"$3h^{\pm}$" : cleaned_arr_simple("TauJets.beamSpotWeight", sel_3p0n, N_ENTRIES).flatten(),
        r"$3h^{\pm}\geq1\pi^{0}$" : cleaned_arr_simple("TauJets.beamSpotWeight", sel_3pXn, N_ENTRIES).flatten(),
    }
    
    """
    red    = ROOT.TColor.GetColor(224, 61, 59)
    blue   = ROOT.TColor.GetColor(62, 88, 189)
    green  = ROOT.TColor.GetColor(34, 140, 121)
    purple = ROOT.TColor.GetColor(137, 90, 145)
    orange = ROOT.TColor.GetColor(245, 184, 37)
    """

    colors = {
        r"$h^{\pm}$" : '#228C79',
        r"$h^{\pm}\pi^{0}$" : '#3E58BD',
        r"$h^{\pm}\geq2\pi^{0}$" : '#E03D3B',
        r"$3h^{\pm}$" : '#885A91',
        r"$3h^{\pm}\geq1\pi^{0}$" : '#F5B825',
    }

    for key in x__var.keys():
        # plot the leading object
        x__var[key] = (x__var[key])[:,0,:].flatten()
        idx = np.nonzero(x__var[key])
        # remove zeros (make sense for variables that is close to zero? yes, this is exactly zero)
        x__var[key] = x__var[key][idx]
        x__weight[key] = x__weight[key][idx]
        if log:
            x__var[key] = np.log10(x__var[key])
        x__weight[key] = x__weight[key] / np.sum(x__weight[key])

    for key in x__weight.keys():
        assert(x__var[key].shape == x__weight[key].shape)

    fig, ax = plt.subplots(figsize=(8, 5))
    plt.subplots_adjust(left=0.12, bottom=0.15, right=0.95, top=0.90)

    for key in x__var.keys():
        x, bins, p=plt.hist(
            x__var[key],
            bins=nbins,
            range=xrange,
            weights=x__weight[key],
            histtype="step", 
            linewidth=1, 
            alpha=1, 
            color=colors[key], 
            label=key, 
        )

    plt.xlabel(varname_tex, fontsize=18)
    plt.ylabel("Arbitrary Unit", fontsize=18)
    
    ax.xaxis.set_minor_locator(AutoMinorLocator())
    ax.yaxis.set_minor_locator(AutoMinorLocator())
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)

    plt.xlim(xrange)

    plt.legend(
        bbox_to_anchor=(0,1.02,1,0.2), 
        loc="lower left", 
        mode="expand", 
        borderaxespad=0, 
        ncol=5, fontsize=14
    )

    plt.savefig(output_name)
    plt.close()


# probably correlations


#region event_display
# pick some events to display
# draw_event_display(sel_1p0n, 10, "scatter_sel_1p0n.pdf", r"Generated $h^{\pm}$")
# draw_event_display(sel_1p1n, 10, "scatter_sel_1p1n.pdf", r"Generated $h^{\pm}\pi^{0}$")
# draw_event_display(sel_1pXn, 18, "scatter_sel_1pXn.pdf", r"Generated $h^{\pm}\geq2\pi^{0}$")
# draw_event_display(sel_3p0n, 16, "scatter_sel_3p0n.pdf", r"Generated $3h^{\pm}$")
# draw_event_display(sel_3pXn, 12, "scatter_sel_3pXn.pdf", r"Generated $3h^{\pm}\geq1\pi^{0}$")

# draw_event_display(sel_1p0n, 10, "scatter_sel_1p0n.extrapolateECal.pdf", r"Generated $h^{\pm}$", True)
# draw_event_display(sel_1p1n, 10, "scatter_sel_1p1n.extrapolateECal.pdf", r"Generated $h^{\pm}\pi^{0}$", True)
# draw_event_display(sel_1pXn, 18, "scatter_sel_1pXn.extrapolateECal.pdf", r"Generated $h^{\pm}\geq2\pi^{0}$", True)
# draw_event_display(sel_3p0n, 16, "scatter_sel_3p0n.extrapolateECal.pdf", r"Generated $3h^{\pm}$", True)
# draw_event_display(sel_3pXn, 12, "scatter_sel_3pXn.extrapolateECal.pdf", r"Generated $3h^{\pm}\geq1\pi^{0}$", True)
#endregion


#region impact paramters
draw_input_variables(
    "TauTrack.d0TJVA", 
    r"$\tau_{had}$ track $d_{0}$ [mm]", 
    50, (-0.2, 0.2), 
    "TauTrack.d0TJVA.pdf"
)
draw_input_variables(
    "TauTrack.d0SigTJVA", 
    r"$\tau_{had}$ track $Sig(d_{0})$", 
    50, (-10, 10), 
    "TauTrack.d0SigTJVA.pdf"
)
draw_input_variables(
    "TauTrack.z0sinthetaTJVA", 
    r"$\tau_{had}$ track $z_{0} \sin\theta$ [mm]", 
    50, (-0.5, 0.5), 
    "TauTrack.z0sinthetaTJVA.pdf"
)
draw_input_variables(
    "TauTrack.z0sinthetaSigTJVA", 
    r"$\tau_{had}$ track $Sig(z_{0} \sin\theta)$", 
    50, (-10, 10), 
    "TauTrack.z0sinthetaSigTJVA.pdf"
)
draw_input_variables(
    "ConvTrack.d0TJVA", 
    r"Conversion track $d_{0}$ [mm]", 
    50, (-3.0, 3.0), 
    "ConvTrack.d0TJVA.pdf"
)
draw_input_variables(
    "ConvTrack.d0SigTJVA", 
    r"Conversion track $Sig(d_{0})$", 
    50, (-15, 15), 
    "ConvTrack.d0SigTJVA.pdf"
)
draw_input_variables(
    "ConvTrack.z0sinthetaTJVA", 
    r"Conversion track $z_{0} \sin\theta$ [mm]", 
    50, (-5.0, 5.0), 
    "ConvTrack.z0sinthetaTJVA.pdf"
)
draw_input_variables(
    "ConvTrack.z0sinthetaSigTJVA", 
    r"Conversion track $Sig(z_{0} \sin\theta)$", 
    50, (-15, 15), 
    "ConvTrack.z0sinthetaSigTJVA.pdf"
)
#endregion

#region moments
draw_input_variables(
    "NeutralPFO.FIRST_ETA", 
    r"Neutral PFO $\left\langle \eta \right\rangle$", 
    50, (-3.0, 3.0), 
    "NeutralPFO.FIRST_ETA.pdf"
)
draw_input_variables(
    "NeutralPFO.SECOND_R", 
    r"Neutral PFO $\left\langle r^{2} \right\rangle$", 
    50, (0., 40000.), 
    "NeutralPFO.SECOND_R.pdf"
)
draw_input_variables(
    "NeutralPFO.DELTA_THETA", 
    r"Neutral PFO $\Delta \theta$", 
    50, (-1.0, 1.0), 
    "NeutralPFO.DELTA_THETA.pdf"
)
draw_input_variables(
    "NeutralPFO.CENTER_LAMBDA", 
    r"Neutral PFO $\lambda_{centre}$", 
    50, (0., 1000.), 
    "NeutralPFO.CENTER_LAMBDA.pdf"
)
draw_input_variables(
    "NeutralPFO.LONGITUDINAL", 
    r"Neutral PFO longitudinal", 
    50, (0.0, 1.0), 
    "NeutralPFO.LONGITUDINAL.pdf"
)
draw_input_variables(
    "NeutralPFO.SECOND_ENG_DENS", 
    r"Neutral PFO $\log_{10}(\left\langle \rho^{2} \right\rangle)$", 
    50, (-9., 1.), 
    "NeutralPFO.SECOND_ENG_DENS.pdf", log=True
)
draw_input_variables(
    "NeutralPFO.ENG_FRAC_CORE", 
    r"Neutral PFO $f_{core}$", 
    50, (0.0, 1.0), 
    "NeutralPFO.ENG_FRAC_CORE.pdf"
)
draw_input_variables(
    "NeutralPFO.NPosECells_EM1", 
    r"Neutral PFO $N_{pos,EM1}$", 
    50, (0., 150.), 
    "NeutralPFO.NPosECells_EM1.pdf"
)
draw_input_variables(
    "NeutralPFO.NPosECells_EM2", 
    r"Neutral PFO $N_{pos,EM2}$", 
    50, (0., 100.), 
    "NeutralPFO.NPosECells_EM2.pdf"
)
draw_input_variables(
    "NeutralPFO.energy_EM1", 
    r"Neutral PFO $E_{EM1}$ [MeV]", 
    50, (-5000., 45000.), 
    "NeutralPFO.energy_EM1.pdf"
)
draw_input_variables(
    "NeutralPFO.energy_EM2", 
    r"Neutral PFO $E_{EM2}$ [MeV]", 
    50, (-5000., 95000.), 
    "NeutralPFO.energy_EM2.pdf"
)
draw_input_variables(
    "NeutralPFO.EM1CoreFrac", 
    r"Neutral PFO $f_{core}^{EM1}$", 
    50, (0.0, 1.0), 
    "NeutralPFO.EM1CoreFrac.pdf"
)
draw_input_variables(
    "NeutralPFO.firstEtaWRTClusterPosition_EM1", 
    r"Neutral PFO $\left\langle \eta_{EM1} \right\rangle$ w.r.t. cluster", 
    50, (-0.03, 0.03), 
    "NeutralPFO.firstEtaWRTClusterPosition_EM1.pdf"
)
draw_input_variables(
    "NeutralPFO.firstEtaWRTClusterPosition_EM2", 
    r"Neutral PFO $\left\langle \eta_{EM2} \right\rangle$ w.r.t. cluster", 
    50, (-0.05, 0.05), 
    "NeutralPFO.firstEtaWRTClusterPosition_EM2.pdf"
)
draw_input_variables(
    "NeutralPFO.secondEtaWRTClusterPosition_EM1", 
    r"Neutral PFO $\left\langle \eta^{2}_{EM1} \right\rangle$ w.r.t. cluster", 
    50, (0, 0.005), 
    "NeutralPFO.secondEtaWRTClusterPosition_EM1.pdf"
)
draw_input_variables(
    "NeutralPFO.secondEtaWRTClusterPosition_EM2", 
    r"Neutral PFO $\left\langle \eta^{2}_{EM2} \right\rangle$ w.r.t. cluster", 
    50, (0, 0.005), 
    "NeutralPFO.secondEtaWRTClusterPosition_EM2.pdf"
)
#endregion

#region input_variables 
# draw_input_variables(
#     "TauTrack.dphiECal", 
#     "TauTrack.dphiECal", 
#     50, (-0.4, 0.4), 
#     "TauTrack.dphiECal.png"
# )
# draw_input_variables(
#     "TauTrack.dphi", 
#     "TauTrack.dphi", 
#     50, (-0.4, 0.4), 
#     "TauTrack.dphi.png"
# )
# draw_input_variables(
#     "TauTrack.detaECal", 
#     "TauTrack.detaECal", 
#     50, (-0.4, 0.4), 
#     "TauTrack.detaECal.png"
# )
# draw_input_variables(
#     "TauTrack.deta", 
#     "TauTrack.deta", 
#     50, (-0.4, 0.4), 
#     "TauTrack.deta.png"
# )
# draw_input_variables(
#     "TauTrack.pt", 
#     "TauTrack.pt", 
#     50, (1000, 81000), 
#     "TauTrack.pt.png"
# )
# draw_input_variables(
#     "TauTrack.jetpt", 
#     "TauTrack.jetpt", 
#     50, (10000, 250000), 
#     "TauTrack.jetpt.png"
# )
# draw_input_variables(
#     "NeutralPFO.dphiECal", 
#     "NeutralPFO.dphiECal", 
#     50, (-0.4, 0.4), 
#     "NeutralPFO.dphiECal.png"
# )
# draw_input_variables(
#     "NeutralPFO.dphi", 
#     "NeutralPFO.dphi", 
#     50, (-0.4, 0.4), 
#     "NeutralPFO.dphi.png"
# )
# draw_input_variables(
#     "NeutralPFO.detaECal", 
#     "NeutralPFO.detaECal", 
#     50, (-0.4, 0.4), 
#     "NeutralPFO.detaECal.png"
# )
# draw_input_variables(
#     "NeutralPFO.deta", 
#     "NeutralPFO.deta", 
#     50, (-0.4, 0.4), 
#     "NeutralPFO.deta.png"
# )
# draw_input_variables(
#     "NeutralPFO.pt", 
#     "NeutralPFO.pt", 
#     50, (1000, 81000), 
#     "NeutralPFO.pt.png"
# )
# draw_input_variables(
#     "NeutralPFO.jetpt", 
#     "NeutralPFO.jetpt", 
#     50, (10000, 250000), 
#     "NeutralPFO.jetpt.png"
# )
# draw_input_variables(
#     "ShotPFO.dphiECal", 
#     "ShotPFO.dphiECal", 
#     50, (-0.4, 0.4), 
#     "ShotPFO.dphiECal.png"
# )
# draw_input_variables(
#     "ShotPFO.dphi", 
#     "ShotPFO.dphi", 
#     50, (-0.4, 0.4), 
#     "ShotPFO.dphi.png"
# )
# draw_input_variables(
#     "ShotPFO.detaECal", 
#     "ShotPFO.detaECal", 
#     50, (-0.4, 0.4), 
#     "ShotPFO.detaECal.png"
# )
# draw_input_variables(
#     "ShotPFO.deta", 
#     "ShotPFO.deta", 
#     50, (-0.4, 0.4), 
#     "ShotPFO.deta.png"
# )
# draw_input_variables(
#     "ShotPFO.pt", 
#     "ShotPFO.pt", 
#     50, (1000, 41000), 
#     "ShotPFO.pt.png"
# )
# draw_input_variables(
#     "ShotPFO.jetpt", 
#     "ShotPFO.jetpt", 
#     50, (10000, 250000), 
#     "ShotPFO.jetpt.png"
# )
# draw_input_variables(
#     "ConvTrack.dphiECal", 
#     "ConvTrack.dphiECal", 
#     50, (-0.4, 0.4), 
#     "ConvTrack.dphiECal.png"
# )
# draw_input_variables(
#     "ConvTrack.dphi", 
#     "ConvTrack.dphi", 
#     50, (-0.4, 0.4), 
#     "ConvTrack.dphi.png"
# )
# draw_input_variables(
#     "ConvTrack.detaECal", 
#     "ConvTrack.detaECal", 
#     50, (-0.4, 0.4), 
#     "ConvTrack.detaECal.png"
# )
# draw_input_variables(
#     "ConvTrack.deta", 
#     "ConvTrack.deta", 
#     50, (-0.4, 0.4), 
#     "ConvTrack.deta.png"
# )
# draw_input_variables(
#     "ConvTrack.pt", 
#     "ConvTrack.pt", 
#     50, (1000, 81000), 
#     "ConvTrack.pt.png"
# )
# draw_input_variables(
#     "ConvTrack.jetpt", 
#     "ConvTrack.jetpt", 
#     50, (10000, 250000), 
#     "ConvTrack.jetpt.png"
# )
#endregion
