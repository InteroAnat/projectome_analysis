(;
    a=[
        (; mac="nmt|charm|Frontal|motor|M1/PM|M1", mus="allen|root|grey|CH|CTX|CTXpl|Isocortex|MO|MOp")
    ],
    b=[
        (; mus="allen|root|grey|BS|MB|MBsta|RAmb|DR", mac="nmt|sarm|mes|Mid|MMid|raphe_Mid|DR")
    ],
    c=[
        (; mac="nmt|sarm|mes|Mid", mus="allen|root|grey|BS|MB")
    ],



    cortex1=[
        ### (; mac=("nmt|charm|Occipital", "nmt|charm|Parietal|SPL|V6/V6A|V6", "nmt|charm|Parietal|SPL|V6/V6A|V6A"), mus=("allen|root|grey|CH|CTX|CTXpl|Isocortex|VIS", "allen|root|grey|CH|CTX|CTXpl|Isocortex|PTLp|VISrl"));

        (; mac="nmt|charm", mus="allen|root|grey|CH|CTX|CTXpl|Isocortex");
        (; mac="nmt|charm|Occipital", mus="allen|root|grey|CH|CTX|CTXpl|Isocortex|VIS")
    ],

    cortex2=[
        (; mac="nmt|charm|Occipital|V1", mus="allen|root|grey|CH|CTX|CTXpl|Isocortex|VIS|VISp");
        (; mac="nmt|charm|Frontal|motor", mus="allen|root|grey|CH|CTX|CTXpl|Isocortex|MO")
    ],

    cortex3=[
        (; mac="nmt|charm|Frontal|ACgG|ACC", mus="allen|root|grey|CH|CTX|CTXpl|Isocortex|ACA");

        (; mac="nmt|charm|Parietal|SI/SII|SI", mus="allen|root|grey|CH|CTX|CTXpl|Isocortex|SS|SSp");
        (; mac="nmt|charm|Parietal|SI/SII|SII", mus="allen|root|grey|CH|CTX|CTXpl|Isocortex|SS|SSs");

        (; mac="nmt|charm|Temporal|ITC|TE", mus="allen|root|grey|CH|CTX|CTXpl|Isocortex|TEa")
    ],

    cortex4=[
        ### (; mac=("nmt|charm|Frontal|motor|M1/PM|PM", "nmt|charm|Frontal|motor|SMA/preSMA|SMA"), mus=("allen|root|grey|CH|CTX|CTXpl|Isocortex|MO|MOs", "allen|root|grey|CH|CTX|CTXpl|Isocortex|PL"));
        
        (; mac="nmt|charm|Frontal|motor|M1/PM|M1", mus="allen|root|grey|CH|CTX|CTXpl|Isocortex|MO|MOp");
        (; mac="nmt|charm|Frontal|OFC|med_OFC|area_10", mus="allen|root|grey|CH|CTX|CTXpl|Isocortex|FRP");

        (; mac="nmt|charm|Temporal|MTL|Rh|ERh", mus="allen|root|grey|CH|CTX|CTXpl|HPF|RHP|ENT");
        (; mac="nmt|charm|Temporal|MTL|Rh|PRh", mus="allen|root|grey|CH|CTX|CTXpl|Isocortex|PERI");
        (; mac="nmt|charm|Temporal|core/belt|core", mus="allen|root|grey|CH|CTX|CTXpl|Isocortex|AUD");
        
        (; mac="nmt|charm|Parietal|PMC|PCgG|RSC", mus="allen|root|grey|CH|CTX|CTXpl|Isocortex|RSP");
        (; mac="nmt|charm|Parietal|IPL|lat_IPS|LIP", mus="allen|root|grey|CH|CTX|CTXpl|Isocortex|PTLp")
    ],

    cortex5=[
        (; mac="nmt|charm|Frontal|OFC|caudal_OFC|OFa-p", mus="allen|root|grey|CH|CTX|CTXpl|Isocortex|AI");
        (; mac="nmt|charm|Frontal|OFC|caudal_OFC|OLF|AON/TTv", mus="allen|root|grey|CH|CTX|CTXpl|OLF|AON");
        (; mac="nmt|charm|Frontal|OFC|caudal_OFC|cl_OFC|G", mus="allen|root|grey|CH|CTX|CTXpl|Isocortex|GU")
    ],




    subcortex2=[
        (; mac="nmt|sarm|di|Hy", mus="allen|root|grey|BS|IB|HY");
        (; mac="nmt|sarm|di|Thal", mus="allen|root|grey|BS|IB|TH");
        (; mac="nmt|sarm|di|EpiThal", mus="allen|root|grey|BS|IB|TH|DORpm|EPI");

        (; mac="nmt|sarm|mes|Mid", mus="allen|root|grey|BS|MB");

        (; mac="nmt|sarm|myel|Med", mus="allen|root|grey|BS|HB|MY");
        (; mac="nmt|sarm|met|Pons", mus="allen|root|grey|BS|HB|P")
    ],


    subcortex3=[
        (; mac="nmt|sarm|tel|BG|Str", mus="allen|root|grey|CH|CNU|STR");
        (; mac="nmt|sarm|tel|BG|Pd", mus="allen|root|grey|CH|CNU|PAL");
        (; mac="nmt|sarm|tel|DSP|ST", mus="allen|root|grey|CH|CNU|PAL|PALc|BST");
        (; mac="nmt|sarm|tel|MPal|HF", mus="allen|root|grey|CH|CTX|CTXpl|HPF");

        (; mac="nmt|sarm|di|Thal|Rt", mus="allen|root|grey|BS|IB|TH|DORpm|RT")
        
        # (; mac="nmt|sarm|tel|POC|opt", mus="allen|root|fiber tracts|cm|IIn|opt");
        # (; mac="nmt|sarm|tel|MPal|f", mus="allen|root|fiber tracts|mfbs|mfbc|fxs");
    ],


    subcortex4=[
        ### (; mac=("nmt|sarm|tel|LVPal|LPal|Cl", "nmt|sarm|tel|LVPal|VPal|En"), mus="allen|root|grey|CH|CTX|CTXsp|CLA");
        ### (; mac="nmt|sarm|tel|BG|Pd|ac", mus=("allen|root|fiber tracts|cm|In|aco", "allen|root|fiber tracts|mfbs|mfbc|act"));
        ### (; mac="nmt|sarm|tel|BG|Pd|GP", mus=("allen|root|grey|CH|CNU|PAL|PALd|GPe", "allen|root|grey|CH|CNU|PAL|PALd|GPi"));
        ### (; mac="nmt|sarm|di|Thal|MLThal|Re-Rh-Xi", mus=("allen|root|grey|BS|IB|TH|DORpm|ILM|RH", "allen|root|grey|BS|IB|TH|DORpm|MTN|RE"));

        (; mac="nmt|sarm|tel|DSP|SDBR|SDB", mus="allen|root|grey|CH|CNU|PAL|PALm|MSC|NDB");
        (; mac="nmt|sarm|tel|MPal|HF|Hi", mus="allen|root|grey|CH|CTX|CTXpl|HPF|HIP");
        
        (; mac="nmt|sarm|di|Thal|MThal|MD", mus="allen|root|grey|BS|IB|TH|DORpm|MED|MD");
        (; mac="nmt|sarm|di|Thal|MLThal|CM", mus="allen|root|grey|BS|IB|TH|DORpm|ILM|CM");
        (; mac="nmt|sarm|di|Thal|MLThal|IAM", mus="allen|root|grey|BS|IB|TH|DORpm|ATN|IAM");
        (; mac="nmt|sarm|di|Thal|MLThal|IMD", mus="allen|root|grey|BS|IB|TH|DORpm|MED|IMD");
        (; mac="nmt|sarm|di|Thal|PThal|LP", mus="allen|root|grey|BS|IB|TH|DORpm|LAT|LP");
        (; mac="nmt|sarm|di|Thal|GThal|MG", mus="allen|root|grey|BS|IB|TH|DORsm|GENd|MG");

        (; mac="nmt|sarm|met|Pons|DPons|PBC", mus="allen|root|grey|BS|HB|P|P-sen|PB")

        # (; mac="nmt|sarm|di|PrT|PCR|pc", mus="allen|root|fiber tracts|cm|IIIn|pc");
        # (; mac="nmt|sarm|tel|MPal|HF|fi", mus="allen|root|fiber tracts|mfbs|mfbc|fxs|fi");
    ],


    subcortex5=[
        ### (; mac="nmt|sarm|di|Thal|MThal|ILThal|CL-PC", mus=("allen|root|grey|BS|IB|TH|DORpm|ILM|CL", "allen|root|grey|BS|IB|TH|DORpm|ILM|PCN"));
        ### (; mac="nmt|sarm|mes|Mid|DMid|Co|SC", mus=("allen|root|grey|BS|MB|MBmot|SCm", "allen|root|grey|BS|MB|MBsen|SCs"));  ####################
        ### (; mac=("nmt|sarm|tel|BG|Str|DStr|Cd", "nmt|sarm|tel|BG|Str|DStr|Pu"), mus="allen|root|grey|CH|CNU|STR|STRd|CP");
        ### (; mac="nmt|sarm|mes|Mid|VMid|DA_Mid|SN", mus=("allen|root|grey|BS|MB|MBmot|SNr", "allen|root|grey|BS|MB|MBsta|SNc"));  ####################
        ### (; mac="nmt|sarm|di|Thal|MThal|ILThal|CMn-PF", mus=("allen|root|grey|BS|IB|TH|DORpm|ILM|CM", "allen|root|grey|BS|IB|TH|DORpm|ILM|PF"));

        (; mac="nmt|sarm|tel|BG|Str|VStr|Acb", mus="allen|root|grey|CH|CNU|STR|STRv|ACB");
        (; mac="nmt|sarm|tel|Amy|pAmy|lpAmy|BL", mus="allen|root|grey|CH|CTX|CTXsp|BLA");
        (; mac="nmt|sarm|tel|Amy|pAmy|lpAmy|La", mus="allen|root|grey|CH|CTX|CTXsp|LA");
        (; mac="nmt|sarm|tel|Amy|pAmy|vpAmy|BM", mus="allen|root|grey|CH|CTX|CTXsp|BMA");
        (; mac="nmt|sarm|tel|Amy|spAmy|stAmy|Ce", mus="allen|root|grey|CH|CNU|STR|sAMY|CEA");
        (; mac="nmt|sarm|tel|Amy|spAmy|mAmy|Me", mus="allen|root|grey|CH|CNU|STR|sAMY|MEA");


        (; mac="nmt|sarm|di|Hy|THy|VTHy|VMH", mus="allen|root|grey|BS|IB|HY|MEZ|VMH");
        (; mac="nmt|sarm|di|Hy|THy|VTHy|Arc", mus="allen|root|grey|BS|IB|HY|PVZ|ARH");
        (; mac="nmt|sarm|di|Hy|THy|DTHy|LHy", mus="allen|root|grey|BS|IB|HY|LZ|LHA");
        (; mac="nmt|sarm|di|Hy|THy|DTHy|AH", mus="allen|root|grey|BS|IB|HY|MEZ|AHN");
        (; mac="nmt|sarm|di|Hy|PHy|DPHy|PH", mus="allen|root|grey|BS|IB|HY|MEZ|PH");
        (; mac="nmt|sarm|di|Hy|THy|DTHy|SOpt", mus="allen|root|grey|BS|IB|HY|PVZ|SO");
        (; mac="nmt|sarm|di|Hy|THy|DTHy|STh", mus="allen|root|grey|BS|IB|HY|LZ|STN");
        (; mac="nmt|sarm|di|Hy|THy|MTHy|Pa", mus="allen|root|grey|BS|IB|HY|PVZ|PVH");


        (; mac="nmt|sarm|mes|Mid|VMid|DA_Mid|VTA", mus="allen|root|grey|BS|MB|MBmot|VTA");
        (; mac="nmt|sarm|mes|Mid|VMid|DA_Mid|IP", mus="allen|root|grey|BS|MB|MBsta|RAmb|IPN");
        (; mac="nmt|sarm|mes|Mid|VMid|DA_Mid|RF", mus="allen|root|grey|BS|MB|MBmot|RR");
        (; mac="nmt|sarm|mes|Mid|VMid|R+|R", mus="allen|root|grey|BS|MB|MBmot|RN");
        (; mac="nmt|sarm|mes|Mid|LMid|TgMid|CnF", mus="allen|root|grey|BS|MB|MBmot|CUN");
        (; mac="nmt|sarm|mes|Mid|LMid|TgMid|PBG", mus="allen|root|grey|BS|MB|MBsen|PBG");
        (; mac="nmt|sarm|mes|Mid|LMid|TgMid|ATg", mus="allen|root|grey|BS|MB|MBmot|AT");
        (; mac="nmt|sarm|mes|Mid|MMid|raphe_Mid|DR", mus="allen|root|grey|BS|MB|MBsta|RAmb|DR");
        (; mac="nmt|sarm|mes|Mid|DMid|PAGR|PAG", mus="allen|root|grey|BS|MB|MBmot|PAG");
        
        (; mac="nmt|sarm|mes|Mid|DMid|Co|SC", mus=("allen|root|grey|BS|MB|MBmot|SCm"));
        (; mac="nmt|sarm|mes|Mid|VMid|DA_Mid|SN", mus="allen|root|grey|BS|MB|MBsta|SNc")
        

        (; mac="nmt|sarm|myel|Med|IMed|MedMC|Nu6", mus="allen|root|grey|BS|HB|MY|MY-mot|VI");
        (; mac="nmt|sarm|myel|Med|IMed|MedRet|DPGi", mus="allen|root|grey|BS|HB|MY|MY-mot|PGRN|PGRNd");
        (; mac="nmt|sarm|myel|Med|IMed|MedRet|Gi", mus="allen|root|grey|BS|HB|MY|MY-mot|GRN");
        (; mac="nmt|sarm|myel|Med|IMed|MedRaphe|RMg", mus="allen|root|grey|BS|HB|MY|MY-sat|RM");
        (; mac="nmt|sarm|myel|Med|IMed|MedRaphe|ROb", mus="allen|root|grey|BS|HB|MY|MY-sat|RO");
        (; mac="nmt|sarm|myel|Med|DMed|VCC|DVC", mus="allen|root|grey|BS|HB|MY|MY-sen|CN");
        (; mac="nmt|sarm|myel|Med|DMed|VCC|Pr", mus="allen|root|grey|BS|HB|MY|MY-mot|PHY|PRP");
        (; mac="nmt|sarm|myel|Med|DMed|VCC|Ve", mus="allen|root|grey|BS|HB|MY|MY-mot|VNC");
        (; mac="nmt|sarm|myel|Med|DMed|SVC|Sol", mus="allen|root|grey|BS|HB|MY|MY-sen|NTS");
        (; mac="nmt|sarm|myel|Med|VMed|PreCbMed|IO", mus="allen|root|grey|BS|HB|MY|MY-mot|IO");
        
        
        (; mac="nmt|sarm|met|Pons|DPons|LGR|CG", mus="allen|root|grey|BS|HB|P|P-mot|PCG");
        (; mac="nmt|sarm|met|Pons|DPons|LGR|LC-Me5", mus="allen|root|grey|BS|HB|P|P-sat|LC");
        (; mac="nmt|sarm|met|Pons|VPons|retPons|PnC", mus="allen|root|grey|BS|HB|P|P-mot|PRNc");
        (; mac="nmt|sarm|met|Pons|VPons|retPons|SubC", mus="allen|root|grey|BS|HB|P|P-sat|SLC")


        # (; mac="nmt|sarm|di|Thal|PThal|Pul|bsc", mus="allen|root|fiber tracts|cm|IIn|bsc");
        # (; mac="nmt|sarm|tel|BG|Str|DStr|ic", mus="allen|root|fiber tracts|lfbs|cst|int");
        # (; mac="nmt|sarm|met|Pons|LPons|ll+|ll", mus="allen|root|fiber tracts|cm|VIIIn|cVIIIn|ll");
        # (; mac="nmt|sarm|met|Pons|VPons|SO/ml|ml", mus="allen|root|fiber tracts|cm|drt|cett|ml");
        # (; mac="nmt|sarm|met|Pons|DPons|PBC|scp", mus="allen|root|fiber tracts|cbf|cbp|scp");
    ],


    subcortex6=[
        (; mac="nmt|sarm|di|Hy|PHy|VPHy|MM/RM|MM", mus="allen|root|grey|BS|IB|HY|MEZ|MBO|MM");

        (; mac="nmt|sarm|di|Thal|MThal|ILThal|SPFC|SPFPC", mus="allen|root|grey|BS|IB|TH|DORsm|SPF");

        (; mac="nmt|sarm|mes|Mid|DMid|Co|ICoC|ICo", mus="allen|root|grey|BS|MB|MBsen|IC");

        (; mac="nmt|sarm|myel|Med|DMed|C5|Sp5|Sp5C", mus="allen|root|grey|BS|HB|MY|MY-sen|SPVC")

        # ?
        (; mac="nmt|sarm|di|Thal|VThal|PNThal|VA", mus="allen|root|grey|BS|IB|TH|DORsm|VENT|VAL")

        # (; mac="nmt|sarm|met|Pons|VPons|Pn+|spf|mcp", mus="allen|root|fiber tracts|cbf|cbp|mcp");
    ],
)
