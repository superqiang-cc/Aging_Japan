# Project: HTTK-Japan
# Project start time: 2023/11/18
# Author: Dr.GUO Qiang, The University of Tokyo
# Contact: qiang@rainbow.iis.u-tokyo.ac.jp
# Description:
# This script is used to test the GAM model for HT-EADs


rm(list=ls())

# load libraries
library(reticulate)
library(lubridate)
library(MASS)
library(ggplot2)
library(patchwork)
library(mgcv)

np <- import("numpy")
input_dir <- "C:/Users/gq646/Desktop/GAM/Aging_Japan/"

# load the input data
date_array_summer <- np$load(paste0(input_dir, "model_input.npz"))$f[["date_array_summer"]]  # (1220, 6)
social_economic <- np$load(paste0(input_dir, "model_input.npz"))$f[["social_economic"]]  # (47, 3, 1220)   population, old, income

jp_hsi_mean_summer <- np$load(paste0(input_dir, "model_input.npz"))$f[["jp_hsi_mean_summer"]]  # (9, 47, 1220) 
jp_hsi_mean_summer_1 <- np$load(paste0(input_dir, "model_input.npz"))$f[["jp_hsi_mean_summer_1"]]  # (9, 47, 1220) 
jp_hsi_mean_summer_2 <- np$load(paste0(input_dir, "model_input.npz"))$f[["jp_hsi_mean_summer_2"]]  # (9, 47, 1220) 

jp_clm_mean_summer <- np$load(paste0(input_dir, "model_input.npz"))$f[["jp_clm_mean_summer"]]  # (5, 47, 1220) 'dewt', 'at', 'rad', 'ws', 'pres' 
jp_clm_mean_summer_1 <- np$load(paste0(input_dir, "model_input.npz"))$f[["jp_clm_mean_summer_1"]]  # (5, 47, 1220) 'dewt', 'at', 'rad', 'ws', 'pres' 
jp_clm_mean_summer_2 <- np$load(paste0(input_dir, "model_input.npz"))$f[["jp_clm_mean_summer_2"]]  # (5, 47, 1220) 'dewt', 'at', 'rad', 'ws', 'pres' 


# load the output data
age_htk <- np$load(paste0(input_dir, "model_input.npz"))$f[["age_htk"]]  # (6, 47, 1220)  'Newborn', 'Baby', 'Teenager', 'Adult', 'Elderly', 'Unclear'
old_htk <- age_htk[5,,]
young_htk <- age_htk[1,,] + age_htk[2,,] + age_htk[3,,] + age_htk[4,,]


# pre-define the array for the predictions
old_prediction <- array(NA, dim=c(10, 10, 47, 1220))
young_prediction <- array(NA, dim=c(10, 10, 47, 1220))

k_sum = 9

# Loop for 47 prefectures
for (pf in seq(47)){
  
  pf_data <- data.frame(
    year = date_array_summer[,1],
    month = date_array_summer[,2],
    day = date_array_summer[,3],
    dow = date_array_summer[,4],
    holiday = date_array_summer[,5],
    dfg = date_array_summer[,6],
    young_population = social_economic[pf, 1,] - social_economic[pf, 2,],
    old_population = social_economic[pf, 2,], 
    income = social_economic[pf, 3,],
    heatstroke_young = young_htk[pf,],
    heatstroke_old = old_htk[pf,]
    )

  
  # for WBGT
  for (hsi in seq(4, 4)){
    
    pf_data$hsi_value <- jp_hsi_mean_summer[hsi, pf,]
    pf_data$hsi_value_1 <- jp_hsi_mean_summer_1[hsi, pf,]
    pf_data$hsi_value_2 <- jp_hsi_mean_summer_2[hsi, pf,]
    
    # cross-validation
    for (yy in seq(10)){
      
      vali_idx <- which(date_array_summer[,1] == yy + 2009)
      cali_idx <- which(date_array_summer[, 1] != yy + 2009)
      
      # young
      mod_young_hsi = gam(heatstroke_young ~ s(hsi_value, k=k_sum) + s(hsi_value_1, k=k_sum) + s(hsi_value_2, k=k_sum) 
                                            + s(income, k=3) + dfg + factor(holiday) + factor(dow) + offset(log(young_population)),
                                            data = pf_data[cali_idx,], family = poisson(), method = "REML", na.action = "na.omit")
      
      young_prediction[yy, hsi, pf, cali_idx] <- predict(mod_young_hsi, newdata = pf_data[cali_idx,], type = "response")
      young_prediction[yy, hsi, pf, vali_idx] <- predict(mod_young_hsi, newdata = pf_data[vali_idx,], type = "response")
      
      # old
      mod_old_hsi = gam(heatstroke_old ~ s(hsi_value, k=k_sum)  + s(hsi_value_1, k=k_sum) + s(hsi_value_2, k=k_sum)
                                        + s(income, k=3) + dfg + factor(holiday) + factor(dow) + offset(log(old_population)),
                                        data = pf_data[cali_idx,], family = poisson(), method = "REML", na.action = "na.omit")
      
      old_prediction[yy, hsi, pf, cali_idx] <- predict(mod_old_hsi, newdata = pf_data[cali_idx,], type = "response")
      old_prediction[yy, hsi, pf, vali_idx] <- predict(mod_old_hsi, newdata = pf_data[vali_idx,], type = "response")
      
      
      print(paste0("Prefecture ", pf, " HSI ", hsi, " Year ", yy + 2009, " Finished."))
      
    }
    
    
  }
  
  
  # for multiple climate variables
  
  pf_data$tair <- jp_clm_mean_summer[2, pf,]
  pf_data$dewt <- jp_clm_mean_summer[1, pf,]
  pf_data$rad <- jp_clm_mean_summer[3, pf,]
  pf_data$ws <- jp_clm_mean_summer[4, pf,]
  pf_data$pres <- jp_clm_mean_summer[5, pf,]
  
  pf_data$tair_1 <- jp_clm_mean_summer_1[2, pf,]
  pf_data$dewt_1 <- jp_clm_mean_summer_1[1, pf,]
  pf_data$rad_1 <- jp_clm_mean_summer_1[3, pf,]
  pf_data$ws_1 <- jp_clm_mean_summer_1[4, pf,]
  pf_data$pres_1 <- jp_clm_mean_summer_1[5, pf,]
  
  pf_data$tair_2 <- jp_clm_mean_summer_2[2, pf,]
  pf_data$dewt_2 <- jp_clm_mean_summer_2[1, pf,]
  pf_data$rad_2 <- jp_clm_mean_summer_2[3, pf,]
  pf_data$ws_2 <- jp_clm_mean_summer_2[4, pf,]
  pf_data$pres_2 <- jp_clm_mean_summer_2[5, pf,]
  
  
  # cross-validation
  for (yy in seq(10)){
    
    vali_idx <- which(date_array_summer[,1] == yy + 2009)
    cali_idx <- which(date_array_summer[, 1] != yy + 2009)
  
    # young
    mod_young_clm = gam(heatstroke_young ~ s(tair, k=k_sum) + s(dewt, k=k_sum) + s(rad, k=k_sum) + s(ws, k=k_sum) + s(pres, k=k_sum) 
                                          + s(tair_1, k=k_sum) + s(dewt_1, k=k_sum) + s(rad_1, k=k_sum) + s(ws_1, k=k_sum) + s(pres_1, k=k_sum) 
                                          + s(tair_2, k=k_sum) + s(dewt_2, k=k_sum) + s(rad_2, k=k_sum) + s(ws_2, k=k_sum) + s(pres_2, k=k_sum) 
                                          + s(income, k=3) + dfg + factor(holiday) + factor(dow)  + offset(log(young_population)), 
                                          data = pf_data[cali_idx,], family = poisson(), method = "REML", na.action = "na.omit")
    
    
    young_prediction[yy, 10, pf, cali_idx] <- predict(mod_young_clm, newdata = pf_data[cali_idx,], type = "response")
    young_prediction[yy, 10, pf, vali_idx] <- predict(mod_young_clm, newdata = pf_data[vali_idx,], type = "response")
    
    # old
    mod_old_clm = gam(heatstroke_old ~ s(tair, k=k_sum) + s(dewt, k=k_sum) + s(rad, k=k_sum) + s(ws, k=k_sum) + s(pres, k=k_sum)
                                    + s(tair_1, k=k_sum) + s(dewt_1, k=k_sum) + s(rad_1, k=k_sum) + s(ws_1, k=k_sum) + s(pres_1, k=k_sum) 
                                    + s(tair_2, k=k_sum) + s(dewt_2, k=k_sum) + s(rad_2, k=k_sum) + s(ws_2, k=k_sum) + s(pres_2, k=k_sum) 
                                    + s(income, k=3) + dfg + factor(holiday) + factor(dow)  + offset(log(old_population)), 
                                    data = pf_data[cali_idx,], family = poisson(), method = "REML", na.action = "na.omit")
    
    
    old_prediction[yy, 10, pf, cali_idx] <- predict(mod_old_clm, newdata = pf_data[cali_idx,], type = "response")
    old_prediction[yy, 10, pf, vali_idx] <- predict(mod_old_clm, newdata = pf_data[vali_idx,], type = "response")
    
    
    print(paste0("Prefecture ", pf, " CLM Year ", yy + 2009, " Finished."))
  
  
  }

}

save(old_prediction, young_prediction,
     file=paste0(input_dir, "Old_Young_Prediction_10y_mean_k9_sp0_offset_dfg_lag3.Rdata"))

print("All Finished.")







