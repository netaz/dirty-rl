# dirty-rl
Learning from making mistakes implementing RL in PyTorch. 

## Tips
- Run the random policy and see how it behaves on your task.  
This will give you a baseline for detecting when your implementation is really 
learning, and when it only appears to be learning.
For example, a random policy controlling an Atari Pong game, looks like this:
[image]

-  Look at the output of the pre-processing.  I messed up the pre-processing in various
ways, and that prevents all learning.  This is usually easy to catch and fix because 
this is a functional bug.  In a burst of wasted energy I wrote a decorator 
`utils.func_footer` which performs a specified action at the exit of the decorated 
function.  I decorated `pre_process_game_frame` so that every 500 input frames, the
pre-processed frame is displayed.  A simple `if` statement would suffice, but would require
an ugly global to count the number of times `pre_process_game_frame` was invoked.  The
extra complexity I introduced is not warranted, but it was fun.
    ```
    @func_footer(500, 
                 lambda processed_img: transforms.ToPILImage()(processed_img.cpu() / 255.).show())
    def pre_process_game_frame(I, n_channels, output_shape):
        ...
    ```

- Watch the game play by rendering the game frames. This is a sanity check: if the
 game is supposed to be random, look for random play.  If the agent arrives at a trivial
 solution, like always moving up, then you should see that.

- If rendering is too expensive, but you want to peak every once in a while and examine
how the agent plays, add interactive rendering: allow keyboard or file to toggle environment 
rendering.

- Add instrumentation: use TensorBoard to log and monitor interesting data like loss and 
episode rewards.  Returns are very noisy in RL from my experience, so use a smoothing 
function like an exponential-moving-average.

- In DQN, we explicitly control the exploration using an epsilon-greedy schedule.  In VPG,
we add stochasticity by sampling an action from a `torch.Categorical` distribution.  We sample
once for each step in the rollout trajectory, and we use the same action-policy for several episodes,
so we can compute the entropy of this distribution.  We should expect to see high entropy (i.e. 
random-ish actions) at the beginning, which decays as the policy learns.

- Speaking of entropy, I followed John Schulman's [advice](https://www.youtube.com/watch?v=jmMsNQ2eug4)
 to maximize initial entropy by initializing the weights of the last linear layer (of only
 2 layers ;-) to zero. The effect on the initial entropy is clear and so is the effect on the 
 performance of the policy.  It's a nice phenomena to see.

## Follow dem gradients
- Gradient accumulation in python.  This comes up in on-policy PG with RL: you collect a batch of 
trajectories (e.g. trajectories collected playing one episode),  experienced following policy PI. 
Then you compute the rewards-to-go (link) and policy “loss”.  
Each trajectory-step requires a forward-pass of the policy, and the results are stored in a batch 
memory (to collect experience).  For each policy forward-pass we also need to compute the policy gradients,
this is obvious enough, using `policy_loss.backward()`.  But we only accumulate gradients, we don’t update
the weights yet, because we want to use a stationary-distribution for the action-policy.  
We keep accumulating the gradients (more on that in the next bullet).

- Let’s follow the gradients: sometimes the model’s graph is not trivial.  The network is not learning,
or learned a bit and got stuck in some very local minimum.  And you suspect that not all parameters are trained 
or updated.  Or maybe you’re trying to learn from the PG trajectories you’ve collected.  For each step we store 
the masked log-probability vector. And that is the root of a backward graph.  

    To create a visualization of the backward graph, I use the incredible little [torchviz](https://github.com/szagoruyko/pytorchviz)
     package.  For example, the graph of two steps (iterations) look like this:
    
    [Here I draw a string of backward graphs]
    
    a.	When we perform the backward pass we traverse these graphs and accumulate gradients in the parameter leaves
    b.	So we should see all of the weights connected to this graph.  The leaf is an `AccumulateGradients` node. 
    c.	We cant visualize more than a couple of backward graphs

- There are several ways to keep your eye on the values of your gradients: 
    - You can plot the weights and gradients distribution and histograms - look at the gradient sizes and ranges: 
     do they make sense?
     While debugging VPG I saw insanely large gradients - a definite red light. I also detected periods of no-learning
     (or very slow learning) by noticing that the weights kept the same distribution shape across a period of training;
    - You can access the parameter gradients directly and print their norms; 
    - You can register to callbacks on parameter updates and on operation backward or forward updates.  This gives
    you access to feature-maps (a.k.a. activations) gradients;
    - You can also access the feature-maps in the policy `forward` method.
    - Besides watching the gradients, I also put little traps for anomalous conditions like all-zero gradients. 
    This is not the case of vanishing gradients, but a case of all-zeros in the gradients.
    ```
        for name, param in q_behavior.named_parameters():
            if param.grad.data.abs().sum() == 0:
                debug = True  # Trap
    ```
  - [Exaplain here how all-zero gradients can come to be]

## Multiprocess Vanilla Policy Gradients
In the MP version of VPG, several processes are launched to perform several episodes of experience rollouts.
As in single-process VPG, each process computes (using backprop) and accumulates the policy gradients using the experience memory.
we perform an `all_reduce` operation on the gradients.  This operation can be a `sum-all` or `mean` operation.
After the `all_reduce`, all processes are synchronized and have the same gradients.

Finally, each process can now update its weights using the gradients.  Because all processes started with the same weights,
and used the same gradients, all weights sets remain equal.

    reduce_gradients(policy, args.gradient_reduce, args.batch_size)
    optimizer.step()

The `all_reduce` operation seems wasteful since it uses full-mesh communication: each process sends
its gradients to all other processes (n * n-1 operations).  I tried improving it by doing a simple 
`reduce` operation on the master-node (rank 0), performing the backprop only on the master-node
(saves power), and then broadcasting the weights from the master-node to all other processes.

Because we only perform the backprop on the master-node, we should get better performance
(less data movement, compute and more cache-locality).
I didn't invest too much time in this, but it seems to help only a little.

    reduce_gradients(policy, args.gradient_reduce, args.batch_size)
    if rank == 0:
        optimizer.step()
    # Make sure that all processes pass this point only after the master-node
    # has completed the back-prop. 
    dist.barrier()
    # broadcast back the weights
    for param in policy.parameters():
        dist.broadcast(param.data, 0)


See also [PyTorch distributed communication](https://pytorch.org/tutorials/intermediate/dist_tuto.html#collective-communication)

For objects sharing state across process-boundaries (like the moving-average tracker) I used
multiprocessing.managers.BaseManager which uses proxies to access shared-state.  This is very
convenient, but a simple queue might be more efficient.  
See also [this](https://www.geeksforgeeks.org/multiprocessing-python-set-2/).



## OpenMP

"The OMP_NUM_THREADS environment variable sets the number of threads to use for parallel regions"
```
os.environ["OMP_NUM_THREADS"] = "1"os.environ["OMP_NUM_THREADS"] = "1"
```
- See the [OpenMP documentation](https://www.openmp.org/spec-html/5.0/openmpse50.htmlhttps://www.openmp.org/spec-html/5.0/openmpse50.html).
- This value must be set before importing `numpy`. 
- In [torch.distributed.launch](https://github.com/pytorch/pytorch/blob/master/torch/distributed/launch.py), processes 
are created using a provided environment, so there's no issue with import order.

```
process = subprocess.Popen(cmd, env=current_env)
```