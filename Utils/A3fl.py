import sys

sys.path.append("../")
import torch
import copy

class Attacker:
    def __init__(self,trigger_size,adv_epochs,target_class,trigger_lr,trigger_outter_epochs,
                 dm_adv_K,dm_adv_model_count,noise_loss_lambda,bkd_ratio,channels,im_size):
        self.trigger_size = trigger_size
        self.adv_epochs = adv_epochs
        self.target_class = target_class
        self.trigger_lr = trigger_lr
        self.trigger_outter_epochs = trigger_outter_epochs
        self.dm_adv_K = dm_adv_K
        self.dm_adv_model_count = dm_adv_model_count
        self.noise_loss_lambda = noise_loss_lambda
        self.bkd_ratio = bkd_ratio
        self.channels = channels
        self.im_size = im_size
        self.setup()

    def setup(self):
        self.handcraft_rnds = 0
        self.trigger = torch.ones((1, self.channels, self.im_size, self.im_size), requires_grad=False, device='cpu') * 0.5
        self.mask = torch.zeros_like(self.trigger)
        self.mask[:, :, 2:2 + self.trigger_size, 2:2 + self.trigger_size] = 1
        self.trigger0 = self.trigger.clone()

    def get_adv_model(self, model, dl, trigger, mask):
        adv_model = copy.deepcopy(model)
        adv_model.train()
        ce_loss = torch.nn.CrossEntropyLoss()
        adv_opt = torch.optim.SGD(adv_model.parameters(), lr=0.01, momentum=0.9, weight_decay=5e-4)
        for _ in range(self.adv_epochs):
            for inputs, labels in dl:
                inputs = trigger * mask + (1 - mask) * inputs
                outputs = adv_model(inputs)
                loss = ce_loss(outputs, labels)
                adv_opt.zero_grad()
                loss.backward()
                adv_opt.step()

        sim_sum = 0.
        sim_count = 0.
        cos_loss = torch.nn.CosineSimilarity(dim=0, eps=1e-08)
        for name in dict(adv_model.named_parameters()):
            if 'features' in name and 'weight' in name or 'conv' in name:
                sim_count += 1
                sim_sum += cos_loss(dict(adv_model.named_parameters())[name].grad.reshape(-1), \
                                    dict(model.named_parameters())[name].grad.reshape(-1))
        return adv_model, sim_sum / sim_count

    def search_trigger(self, model, dl):
        model.eval()
        adv_models = []
        adv_ws = []

        ce_loss = torch.nn.CrossEntropyLoss()
        alpha = self.trigger_lr

        K = self.trigger_outter_epochs
        t = self.trigger.clone()
        m = self.mask.clone()
        normal_grad = 0.
        count = 0
        for iter in range(K):
            if iter % self.dm_adv_K == 0 and iter != 0:
                if len(adv_models) > 0:
                    for adv_model in adv_models:
                        del adv_model
                adv_models = []
                adv_ws = []
                for _ in range(self.dm_adv_model_count):
                    adv_model, adv_w = self.get_adv_model(model, dl, t, m)
                    adv_models.append(adv_model)
                    adv_ws.append(adv_w)

            for inputs, labels in dl:
                count += 1
                t.requires_grad_()
                inputs = t * m + (1 - m) * inputs
                labels[:] = self.target_class
                outputs = model(inputs)
                loss = ce_loss(outputs, labels)

                if len(adv_models) > 0:
                    for am_idx in range(len(adv_models)):
                        adv_model = adv_models[am_idx]
                        adv_w = adv_ws[am_idx]
                        outputs = adv_model(inputs)
                        nm_loss = ce_loss(outputs, labels)
                        if loss == None:
                            loss = self.noise_loss_lambda * adv_w * nm_loss / self.dm_adv_model_count
                        else:
                            loss += self.noise_loss_lambda * adv_w * nm_loss / self.dm_adv_model_count
                if loss != None:
                    loss.backward()
                    normal_grad += t.grad.sum()
                    new_t = t - alpha * t.grad.sign()
                    t = new_t.detach_()
                    t = torch.clamp(t, min=-2, max=2)
                    t.requires_grad_()
        t = t.detach()
        self.trigger = t
        self.mask = m

    def poison_input(self, inputs, labels, eval=False):
        poison_indices = (labels == 0)

        inputs[poison_indices] = (
                self.trigger * self.mask +
                inputs[poison_indices] * (1 - self.mask)
        )

        labels[poison_indices] = self.target_class

        return inputs, labels


def RunAttack(net, trainLoader, epochs, global_model,attacker, device, verbose=False, lr=0.01,
                                              round=0, target_label = 1, size = 3):
    """
        Perform a single attack round on the local model.
        """
    attacker.search_trigger(global_model, trainLoader)

    mask = attacker.mask.detach().cpu().clone()
    trigger = attacker.trigger.detach().cpu().clone()
    if verbose:
        print(f"Trigger updated.")

    # Train localnet on poisoned data
    net.train()
    optimizer = torch.optim.SGD(net.parameters(), lr=lr)
    ce_loss = torch.nn.CrossEntropyLoss()

    for epoch in range(epochs):
        net.train()

        running_loss = 0.0
        correct = 0
        total = 0

        for inputs, labels in trainLoader:

            # Poison input batch
            inputs, labels = attacker.poison_input(inputs, labels, eval=False)
            inputs,labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = net(inputs)
            loss = ce_loss(outputs, labels)
            loss.backward()
            optimizer.step()

            # ---- metrics ----
            running_loss += loss.item() * inputs.size(0)
            _, predicted = outputs.max(1)
            correct += predicted.eq(labels).sum().item()
            total += labels.size(0)

        epoch_loss = running_loss / total
        epoch_acc = correct / total

        if verbose:
            print(f"Train: "
                  f"Loss: {epoch_loss:.2f} "
                  f"Accuracy: {epoch_acc * 100:.2f}%")

    return [epoch_loss], [epoch_acc], mask, trigger