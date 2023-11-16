# -*- coding: utf-8 -*-
# @Author: tianl
# @Date:   2021-01-04 14:03:30
# @Last Modified by:   tianl
# @Last Modified time: 2021-01-11 14:05:39

# Plot Atompair Radar Plot on test set 
                                                    

sub_n = "Atompair_"

idx = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9] 

for (i, name) in zip(idx, c_n):

  roc_auc_NB = float(NB_metrics_atom_tests[i][0]) 
  roc_auc_RF = float(RF_metrics_atom_tests[i][0]) 
  roc_auc_LR = float(LR_metrics_atom_tests[i][0]) 
  roc_auc_SVM = float(SVM_metrics_atom_tests[i][0])   
  roc_auc_MLP = float(MLP_metrics_atom_tests[i][0])  
  roc_auc_AD = float(AD_metrics_atom_tests[i][0]) 
  roc_auc_DT = float(DT_metrics_atom_tests[i][0])

  Fl_score_NB = float(Ntmetrics_atoputests[i][1]) 
  Fl_score_RF = float(RF_metrics_atom_tests[i][1]) 
  Fl_score_LR = float(LR_metrics_atom_tests[i][1]) 
  Fl_score_SVM = float(SVCmetrics_atom_tests[i][1]) 
  Fl_score_MLP = float(MLP_metrics_atom_tests[i][1])   
  Fl_score_AD = float(AD_metrics_atom_tests[i][1]) 
  Fl_score_DT = float(DT_metrics_atom_tests[i][1])

  ACC NB = float(NB_metrics_atom_tests[1][2]) 
  ACC_RF = float(RF_metrics_atom_tests[i][2]) 
  ACC_LR = float(LR_metrics_atom_tests[i][2]) 
  ACC_SVM = float(SVM_metrics_atom_tests[i][2]) 
  ACC_MLP = float(MLP_metrics_atom_tests[i][2]) 
  ACC AD = float(AD_metrics_atom_tests[i][2]) 
  ACC DT = float(DT_metrics_atom_tests[i][2])

  Cohan_Kappa_NB = float(NB_metrics_atom_testsa][3])  
  Cohan_Kappa_RF = float(RF_metrics_atom_testsal(3]) 
  Cohan_Kappa_LR = float(LR_metrics_atom_testsE0(3])

    MCC NB = float(Ntmetrics_atom_tests:1) CC) 
    MCCJtF = float(RF_metries_atom_tests:1)[41]) 
    MCC_LR = float(LR_metries_atom_tests:il[4]) 
    MCC_SVM = float(SVM_metrics_atom_testsaliC 
    MCC_MLP = float(MLP_metrics_atom_tests[i][4]) 
    MCC AD = float(p_metrics_atom_tests:il[4]) 
    MCC_DT = float(bT_metrics_atom_tests[i][4])
    
    precision_NB = float(NB_metrics_atom_tests:i][5]) 
    precision_RF = float(RF_metrics_atom_tests:i][5]) 
    precision_LR = float(LR_metrics_atom_tests:i][5]) 
    precision_SVM = float(SVM_metrics_atom_tests[i][5]) precision_MLP = float(MLP_metrics_atom_tests[i][5]) precision_AD = float(a_metrics_atom_tests:i][5]) 
    precision_MLP = float(MLP_metrics_atom_tests[i][5]) precision_AD = float(a_metrics_atom_tests:i][5]) 
    precision_AD = float(a_metrics_atom_tests:i][5]) 
    precision_DT = float(DT_metrics_atom_tests[i][5])
    
    recall_NB = float(NELmetrics_atom_testsEilE6]) 
    recall_RF = float(RF_metrics_atom_testsEil[6]) 
    recall_LR = float(LR_metrics_atom_tests(i][6]) 
    recall_SVM = float(SVM_metries_atom_tests[i][6]) 
    recall_MLP = float(MLP_metrics_atom_tests[i][6]) 
    recall_AD = float(AD_metries_atom_testsalifd) 
    recall_DT = float(Dlimetrics_atom_testsCili6])
    
layout = go.Layout(
autosize=False,
width=1000, 
height=300, 
margin=go.Margin(
1=250,
r=250)

 radar_chart = go.Figure(layout=layout)

 radar_chart.add_trace(
       go.Scatterpolar(
        name = "Logistic Regression"
           r=[roc_auc_LR, F1_score, ACC, Cohan_Kappa, MCC, precision, recall, roc_auc],
           theta=['ROC_AUC', 'F1_Score', 'ACC', 'Cohan_Kappa', 'MCC', 'Precision', 'Recall', 'ROC_AUC'],
           showlegend=True
           )
       )


   radar_chart.update_layout(
   polar=dict(
       radialaxis=dict(
           visible=True,
           range=[0, 1],
           tickfont=dict(size=16)
       ),
       angularaxis = dict(
           tickfont = dict(size = 20)
       ),
   ),

   title="Atompair on test set: " + name, font=dict(
   size=17)

    radar_chart.show()
   radar_chart.write_image("Radar_Chart_test/" + name + ".png")

