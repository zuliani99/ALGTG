
from ta_vaal_query_model import VAE, Discriminator

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset

import numpy as np

from ActiveLearner import ActiveLearner

from typing import Dict, Any, List, Tuple

import logging
logger = logging.getLogger(__name__)


class TA_VAAL(ActiveLearner):
    
    def __init__(self, ct_p: Dict[str, Any], t_p: Dict[str, Any], al_p: Dict[str, Any]) -> None:
        
        super().__init__(ct_p, t_p, al_p, self.__class__.__name__)
        
    def read_data(self, dataloader: DataLoader, labels=True):
        if labels:
            while True:
                for _, img, label in dataloader:
                    yield img, label
        else:
            while True:
                for _, img, _ in dataloader:
                    yield img

        
    def vae_loss(self, x, recon, mu, logvar, beta):
        mse_loss = nn.MSELoss()
        MSE = mse_loss(recon, x)
        KLD = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
        KLD = KLD * beta
        return MSE + KLD
    
    def train_vaal(self, models: Dict[str, VAE | Discriminator], optimizers: Dict[str, optim.Adam], \
                   cycle: int, n_top_k_obs: int, subset_len: int) -> None:
        
        EPOCHV = 120
        vae = models['vae']
        discriminator = models['discriminator']
        
        self.load_best_checkpoint()
        self.model.eval()

        vae.train()
        discriminator.train()
        
        vae = vae.to(self.device)
        discriminator = discriminator.to(self.device)

        adversary_param = 1
        beta          = 1
        num_adv_steps = 1
        num_vae_steps = 1

        bce_loss = nn.BCELoss()
        
        labeled_data = self.read_data(self.lab_train_dl)
        unlabeled_data = self.read_data(self.unlab_train_dl)

        train_iterations = int( (n_top_k_obs * cycle + subset_len) * EPOCHV / self.ds_t_p['batch_size'] )

        for iter_count in range(train_iterations):
            labeled_imgs, labels = next(labeled_data)
            unlabeled_imgs = next(unlabeled_data)[0]

            labeled_imgs = labeled_imgs.to(self.device)
            unlabeled_imgs = unlabeled_imgs.to(self.device)
            labels = labels.to(self.device)
            if iter_count == 0 :
                r_l_0 = torch.from_numpy(np.random.uniform(0, 1, size=(labeled_imgs.shape[0],1))).to(torch.float32).to(self.device)#.cuda()
                r_u_0 = torch.from_numpy(np.random.uniform(0, 1, size=(unlabeled_imgs.shape[0],1))).to(torch.float32).to(self.device)#.cuda()
            else:
                with torch.no_grad():
                    _, _, r_l =self.model(labeled_imgs)
                    _, _, r_u =self.model(unlabeled_imgs)
                    
            if iter_count == 0:
                r_l = r_l_0.detach()
                r_u = r_u_0.detach()
                r_l_s = r_l_0.detach()
                r_u_s = r_u_0.detach()
            else:
                r_l_s = torch.sigmoid(r_l).detach()
                r_u_s = torch.sigmoid(r_u).detach()                 
            # VAE step
            for count in range(num_vae_steps): # num_vae_steps
                recon, _, mu, logvar = vae(r_l_s,labeled_imgs)
                unsup_loss = self.vae_loss(labeled_imgs, recon, mu, logvar, beta)
                unlab_recon, _, unlab_mu, unlab_logvar = vae(r_u_s,unlabeled_imgs)
                transductive_loss = self.vae_loss(unlabeled_imgs, 
                        unlab_recon, unlab_mu, unlab_logvar, beta)
            
                labeled_preds = discriminator(r_l,mu)
                unlabeled_preds = discriminator(r_u,unlab_mu)
                
                lab_real_preds = torch.ones(labeled_imgs.size(0))
                unlab_real_preds = torch.ones(unlabeled_imgs.size(0))
                    
                lab_real_preds = lab_real_preds.to(self.device)
                unlab_real_preds = unlab_real_preds.to(self.device)

                dsc_loss = bce_loss(labeled_preds[:,0], lab_real_preds) + \
                        bce_loss(unlabeled_preds[:,0], unlab_real_preds)
                total_vae_loss = unsup_loss + transductive_loss + adversary_param * dsc_loss
                
                optimizers['vae'].zero_grad()
                total_vae_loss.backward()
                optimizers['vae'].step()

                # sample new batch if needed to train the adversarial network
                if count < (num_vae_steps - 1):
                    labeled_imgs, _ = next(labeled_data)
                    unlabeled_imgs = next(unlabeled_data)[0]

                    labeled_imgs = labeled_imgs.to(self.device)
                    unlabeled_imgs = unlabeled_imgs.to(self.device)
                    labels = labels.to(self.device)

            # Discriminator step
            for count in range(num_adv_steps):
                with torch.no_grad():
                    _, _, mu, _ = vae(r_l_s,labeled_imgs)
                    _, _, unlab_mu, _ = vae(r_u_s,unlabeled_imgs)
                
                labeled_preds = discriminator(r_l,mu)
                unlabeled_preds = discriminator(r_u,unlab_mu)
                
                lab_real_preds = torch.ones(labeled_imgs.size(0))
                unlab_fake_preds = torch.zeros(unlabeled_imgs.size(0))

                lab_real_preds = lab_real_preds.to(self.device)
                unlab_fake_preds = unlab_fake_preds.to(self.device)
                
                dsc_loss = bce_loss(labeled_preds[:,0], lab_real_preds) + \
                        bce_loss(unlabeled_preds[:,0], unlab_fake_preds)

                optimizers['discriminator'].zero_grad()
                dsc_loss.backward()
                optimizers['discriminator'].step()

                # sample new batch if needed to train the adversarial network
                if count < (num_adv_steps-1):
                    labeled_imgs, _ = next(labeled_data)
                    unlabeled_imgs = next(unlabeled_data)[0]

                    labeled_imgs = labeled_imgs.to(self.device)
                    unlabeled_imgs = unlabeled_imgs.to(self.device)
                    labels = labels.to(self.device)
                if iter_count % 100 == 0:
                    logger.info("Iteration: " + str(iter_count) + "  vae_loss: " + str(total_vae_loss.item()) + " dsc_loss: " +str(dsc_loss.item()))

    
    
    def query(self, sample_unlab_subset: Subset, n_top_k_obs: int) -> Tuple[List[int], List[int]]:
            
        # set the entire batch size to the dimension of the sampled unlabeled set
        self.unlab_train_dl = DataLoader(
            sample_unlab_subset, batch_size=self.ds_t_p['batch_size'],
            shuffle=False, pin_memory=True,
        )
        self.lab_train_dl = DataLoader(
            self.labeled_subset, batch_size=self.ds_t_p['batch_size'],
            shuffle=False, pin_memory=True,
        )
        
        self.idxs_unlab = { 'idxs': torch.empty(0, dtype=torch.int8) }
        self.get_embeddings(self.unlab_train_dl, self.idxs_unlab)
        
        vae = VAE()
        discriminator = Discriminator(self.dataset.image_size)
     
        tavaal_models = {'vae': vae, 'discriminator': discriminator}
        
        optim_vae = optim.Adam(vae.parameters(), lr=5e-4)
        optim_discriminator = optim.Adam(discriminator.parameters(), lr=5e-4)
        optimizers = {'vae': optim_vae, 'discriminator': optim_discriminator}

        self.train_vaal(tavaal_models, optimizers, self.iter+1, n_top_k_obs, len(sample_unlab_subset))
              
        all_preds, all_indices = [], []

        for indices, images, _ in self.unlab_train_dl:                       
            images = images.cuda()
            with torch.no_grad():
                _,_,r = self.model(images)              
                _, _, mu, _ = vae(torch.sigmoid(r),images)
                preds = discriminator(r,mu)

            preds = preds.cpu().data
            all_preds.extend(preds)
            all_indices.extend(indices)

        all_preds = torch.stack(all_preds)
        all_preds = all_preds.view(-1)
        # need to multiply by -1 to be able to use torch.topk 
        all_preds *= -1
        # select the topk points which the discriminator things are the most likely to be unlabeled
        overall_topk = torch.topk(all_preds, n_top_k_obs, largest=False)
        
        return overall_topk.indices.tolist(), [int(self.idxs_unlab['idxs'][id].item()) for id in overall_topk.indices.tolist()] 

        
    