# %%
import random
from simpn.simulator import SimProblem, SimToken
from ocpn_prototypes import OCPNVar, OCPNEvent
from ocpn_reporter import OCELReporter

from base import build_model

# %%
sim = build_model()

# %%

# Replace default PO tokens with anomaly-injected ones
# Clear the place first, then re-populate
sim.PO_arrival.marking.clear()
# sim.PO_arrival.remove_token()