# label1
label1_classes = ['detritus', 'plankton']

label1 = range(0, 1)

label1_map = dict(zip(label1, label1_classes))

# label2
label2_classes = ['copepod', 'detritus', 'noncopepod']
label2_classes_only = label2_classes
label2_classes_only.remove('detritus')

label2 = range(0, 1)
label2_detritus = range(0, 3)

label2_detritus_map = dict(zip(label2_detritus, label2_classes))
label2_map = dict(zip(label2, label2_classes_only))

# label3
label3_classes = ['annelida_polychaeta', 'appendicularia', 'bivalvia-larvae', 'byrozoa-larvae', 'chaetognatha',
                  'cirripedia_barnacle-nauplii', 'cladocera', 'cladocera_evadne-spp', 'cnidaria', 'copepod_calanoida',
                  'copepod_calanoida_acartia-spp', 'copepod_calanoida_calanus-spp', 'copepod_calanoida_candacia-spp',
                  'copepod_calanoida_centropages-spp', 'copepod_calanoida_para-pseudocalanus-spp',
                  'copepod_calanoida_temora-spp', 'copepod_cyclopoida', 'copepod_cyclopoida_corycaeus-spp',
                  'copepod_cyclopoida_oithona-spp', 'copepod_cyclopoida_oncaea-spp', 'copepod_harpacticoida',
                  'copepod_nauplii', 'copepod_unknown', 'decapoda-larvae_brachyura',
                  'detritus', 'echniodermata-larvae', 'euphausiid',
                  'euphausiid_nauplii', 'fish-eggs', 'gastropoda-larva',
                  'mysideacea', 'nt-bubbles', 'nt-phyto_ceratium-spp',
                  'nt-phyto_rhizosolenia-spp', 'nt_phyto_chains', 'ostracoda',
                  'radiolaria', 'tintinnida', 'tunicata_doliolida',
                  ]
label3_classes_only = label3_classes
label3_classes_only.remove('detritus')

label3 = range(0, 38)
label3_detritus = range(0, 39)

label3_detritus_map = dict(zip(label3, label3_classes))
label3_map = dict(zip(label3_detritus, label3_classes_only))

# define experiments
experiments = {
    'label1': label1_map,
    'label2': label2_map,
    'label2_detritus': label2_detritus_map,
    'label3': label3_map,
    'label3_detritus': label3_detritus_map
}


class PlanktonLabels:
    def __init__(self, experiment='label3_wdetritus'):
        self.experiment = experiment

    def labels(self):
        self.labels = experiments[self.experiment]
        return self.labels