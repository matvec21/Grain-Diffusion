# %%

import matplotlib.pyplot as plt
from model import *

from tqdm.auto import tqdm
import matplotlib.image as images
import os

# %%

graphs = []
for name in os.listdir('example_dataset'):
    if name[-4:] != '.png' or name[0] != 'z':
        continue
    img = images.imread('example_dataset/' + name)
    if len(img.shape) > 2:
        img = img[:, :, 0]
    img[img >= 150 / 255] = 1
    zerno_graph = np.argmin(img[::-1], axis = 0).astype(np.float64) / img.shape[0]
    graphs.append(zerno_graph)
graphs = np.vstack(graphs)

save_mean = graphs.mean()
save_std = graphs.std()
graphs = (graphs - save_mean) / save_std

graphs = torch.FloatTensor(graphs).unsqueeze(1)

# %%

torch.save((save_mean, save_std), 'mean_std_save.pth')

# %%

fig, axes = plt.subplots(3, 2, figsize=(6, 9))

for i, ax in enumerate(axes.flatten()):
    ax.plot(graphs[i, 0], color = 'r')
    ax.set_title(f'Зерно {i+1}')

plt.tight_layout()
plt.show()

# %%

graphs = graphs.to(device)

# %%

def do_diffusion(data : torch.Tensor):
    data = data.unsqueeze(0).repeat(TIME_STEPS, 1, 1, 1)
    noise = torch.randn(data.shape, device = data.device)
    data = data * sqrt_beta_prod + sqrt_beta_prod_step * noise
    return data, noise

samples, _ = do_diffusion(graphs)

# %%

for t in samples[240, :].cpu():
    plt.plot(t[0], c = 'navy', alpha = 0.25)
plt.title('Noised')
plt.show()

# %%

model = Diffusion1D(80, 32, 8).to(device)

optim = torch.optim.AdamW(
    model.parameters(),
    lr = 1e-3, weight_decay = 1e-6,
)

# %%

print('TOTAL PARAMETERS:', sum(param.numel() for param in model.parameters()))

# %%

batch_size = graphs.shape[0]
t_emb = timestep_embedding(torch.arange(TIME_STEPS, device = graphs.device), model.t_dim).unsqueeze(0).repeat(batch_size, 1, 1).view(TIME_STEPS * batch_size, model.t_dim)

loss_history = []
bar = tqdm(range(4000))

for e in bar:
    optim.zero_grad()

    noised, noise = do_diffusion(graphs)
    pred = model(noised.view(TIME_STEPS * batch_size, 1, -1), t_emb).view(TIME_STEPS, batch_size, 1, -1)

    loss = torch.pow(pred - noise, 2).mean()
    loss.backward()
    optim.step()

    bar.set_description(f'Loss: {loss.item():.4f}')
    loss_history.append(loss.item())

# %%

plt.plot(loss_history)
plt.yscale('log')
plt.ylabel('Loss')
plt.xlabel('Training step')
plt.show()

# %%

model.eval()

with torch.no_grad():
    samps = generate(model, 1)
    samps = samps.cpu()
    for t in samps[:, 0]:
        plt.plot(t, alpha = 1, color = 'red')
    plt.show()

# %%

for t in graphs[:, 0].cpu():
    plt.plot(t, alpha = 0.25, color = 'red')
plt.show()

# %%

torch.save(model.state_dict(), 'model6.pth')

# %%
