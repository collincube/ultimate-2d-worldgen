"""Game-world systems for features 41-60: NPCs, factions, shops, and quests.

Features exchange maps, graphs, and events through the World registries.
They generate fictional gameplay state: guard detection, faction reputation,
resource shortages, alliances, rivalries, and encounter objectives.
"""
from __future__ import annotations

import math
from collections import defaultdict, deque

import numpy as np

import tilegen_fields as tf
import tilegen_world as tw
from tilegen_world import feature

FACTION_NAMES = ['the Salt Assembly', 'House Vareth', 'the Ash Guild',
                 'the Quiet Order', 'the Dockmen', 'the Ninefold Court',
                 'the Kiln Compact', 'the Grey Wardens']
ROLES = ['labourer', 'artisan', 'clerk', 'guard', 'merchant', 'priest',
         'physician', 'magistrate', 'courier', 'steward']
CLASSES = ['underclass', 'labouring', 'artisan', 'mercantile', 'patrician']
GIVEN = ['Ilse', 'Corin', 'Halda', 'Mero', 'Tavish', 'Oria', 'Renn', 'Sabel',
         'Kestrel', 'Dorn', 'Vay', 'Imra', 'Lock', 'Perrin', 'Yael', 'Fen']
FAMILY = ['Vane', 'Ashcroft', 'Bellhow', 'Cray', 'Dunmar', 'Elgin', 'Fother',
          'Garrick', 'Holt', 'Iselm', 'Jarrow', 'Kestermund']


# ------------------------------------------------------------- inhabitants

def ensure_population(world, per_place=1.2):
    """Create NPCs with homes, workplaces, and gameplay attributes on demand."""
    if world.entities:
        return world.entities
    rng = world.rng('population')
    homes = [p for p in world.places.values() if p.kind in ('room', 'plaza')]
    if not homes:
        homes = list(world.places.values())
    if not homes:
        return world.entities
    n = max(6, int(len(homes) * per_place))
    for i in range(n):
        home = homes[int(rng.integers(len(homes)))]
        work = homes[int(rng.integers(len(homes)))]
        eid = tw.entity_id(world.seed, i)
        age = int(rng.integers(18, 74))
        world.entities[eid] = {
            'id': eid,
            'name': f"{GIVEN[int(rng.integers(len(GIVEN)))]} "
                    f"{FAMILY[int(rng.integers(len(FAMILY)))]}",
            'age': age,
            'role': ROLES[int(rng.integers(len(ROLES)))],
            'social_class': CLASSES[int(rng.integers(len(CLASSES)))],
            'home': home.id,
            'work': work.id,
            'faction': None,
            'reputation': round(float(rng.random()), 3),
            'wealth': round(float(rng.random()), 3),
            'health': round(float(rng.random() * 0.4 + 0.6), 3),
            'morale': round(float(rng.random() * 0.5 + 0.4), 3),
        }
    return world.entities


def _entities_at(world, place_id, key='home'):
    return [e for e in world.entities.values() if e.get(key) == place_id]


# ----------------------------------------------- 41 Territory & Sovereignty

@feature('soc.sovereignty', 41, 'Faction Territories',
         'Society & Institutions', consumes=('places',),
         produces=('fields.control', 'artifacts.factions',
                   'topology.institutions.jurisdiction'), part='III')
def sovereignty(world):
    """Factions claim places; control decays with distance from a seat.

    Contested ground is where two claims are close in strength. These values
    feed guard response, market, and combat encounter fields.
    """
    ensure_population(world)
    rng = world.rng('sovereignty')
    adjacency = world.topology('physical.adjacency')
    candidates = sorted(world.places.values(), key=lambda p: -p.area)
    nfac = max(2, min(len(FACTION_NAMES), max(2, len(candidates) // 8)))
    seats = candidates[:nfac]

    for i, seat in enumerate(seats):
        fid = tw.faction_id(world.seed, i)
        world.factions[fid] = {
            'id': fid, 'name': FACTION_NAMES[i % len(FACTION_NAMES)],
            'seat': seat.id, 'legitimacy': round(float(rng.random() * 0.5 + 0.4), 3),
            'militancy': round(float(rng.random()), 3),
            'wealth': round(float(rng.random()), 3),
            'territory': [], 'members': [],
        }
        seat.tags.add('seat of power')

    strength = {}
    for fid, fac in world.factions.items():
        dist = adjacency.shortest_paths(fac['seat'])
        span = max(dist.values()) if dist else 1.0
        strength[fid] = {pid: max(0.0, 1.0 - d / max(span, 1e-6))
                         for pid, d in dist.items()}

    control_owner, control_value, contested = {}, {}, {}
    for pid in world.place_order:
        claims = sorted(((strength[f].get(pid, 0.0), f) for f in world.factions),
                        reverse=True)
        if not claims or claims[0][0] <= 0:
            continue
        top, owner = claims[0]
        second = claims[1][0] if len(claims) > 1 else 0.0
        control_owner[pid] = owner
        control_value[pid] = round(float(top), 3)
        contested[pid] = round(float(1.0 - (top - second)), 3)
        world.factions[owner]['territory'].append(pid)
        world.places[pid].meta['sovereign'] = owner
        if contested[pid] > 0.85:
            world.places[pid].tags.add('contested')

    for e in world.entities.values():
        owner = control_owner.get(e['home'])
        if owner:
            e['faction'] = owner
            world.factions[owner]['members'].append(e['id'])

    world.set_field('control', world.place_mask_field(control_value))
    world.set_field('contested', world.place_mask_field(contested))

    jur = world.topology('institutions.jurisdiction')
    for pid, owner in control_owner.items():
        for nb in world.places[pid].neighbours:
            if control_owner.get(nb) == owner:
                jur.add(pid, nb, 1.0)
    world.artifacts['factions'] = {
        f: {k: (len(v) if isinstance(v, list) else v) for k, v in fac.items()}
        for f, fac in world.factions.items()}
    return world.factions


# --------------------------------------------------- 42 Political Influence

@feature('soc.politics', 42, 'Faction Influence and Rivalries',
         'Society & Institutions', consumes=('artifacts.factions',),
         produces=('artifacts.politics', 'topology.society.influence'), part='III')
def politics(world):
    """Who owes whom. Open influence between factions, plus a secret layer that
    deliberately crosses the public alignments."""
    rng = world.rng('politics')
    infl = world.topology('society.influence', directed=True)
    fids = list(world.factions)
    for a in fids:
        for b in fids:
            if a == b:
                continue
            w = float(rng.random())
            if w > 0.55:
                infl.add(a, b, round(w, 3))

    conspiracies = []
    if len(fids) >= 2:
        for i in range(max(1, len(fids) // 2)):
            plotters = list(rng.choice(fids, size=min(2, len(fids)), replace=False))
            target = fids[int(rng.integers(len(fids)))]
            if target in plotters:
                continue
            conspiracies.append({
                'id': f'cns_{i}',
                'plotters': plotters, 'target': target,
                'goal': ['recover an artifact', 'rescue a scout',
                         'disable a barrier', 'explore a dungeon'][i % 4],
                'secrecy': round(float(rng.random() * 0.5 + 0.5), 3),
                'exposure_risk': round(float(rng.random() * 0.4), 3)})
    leverage = {}
    for a in fids:
        reach = infl.shortest_paths(a)
        leverage[a] = round(len(reach) / max(len(fids), 1), 3)
    world.artifacts['politics'] = {'influence_edges': infl.edge_count(),
                                   'conspiracies': conspiracies,
                                   'leverage': leverage}
    return conspiracies


# ------------------------------------------------------------------ 43 Law

OFFENCES = {
    'trespass': 0.2, 'theft': 0.4, 'smuggling': 0.5, 'assault': 0.7,
    'sedition': 0.8, 'murder': 1.0, 'blasphemy': 0.5, 'fraud': 0.45,
}


@feature('soc.law', 43, 'Town Rules and Guard Responses', 'Society & Institutions',
         consumes=('artifacts.factions', 'fields.guard_attention'),
         produces=('artifacts.law', 'fields.enforcement'), part='III')
def law(world):
    """Jurisdiction, an offence catalogue, and enforcement that varies by place.

    Severity values range from 0 to 1 and vary with faction militancy.
    Enforcement combines guard attention with the controlling faction's
    legitimacy to give each location a guard-response value.
    """
    rng = world.rng('law')
    attention = world.field('guard_attention', 0.2)
    codes = {}
    for fid, fac in world.factions.items():
        severity = {}
        for off, base in OFFENCES.items():
            severity[off] = round(float(np.clip(
                base * (0.6 + fac['militancy'] * 0.8) +
                rng.normal(0, 0.05), 0, 1)), 3)
        codes[fid] = {'faction': fac['name'], 'severity': severity,
                      'due_process': round(float(fac['legitimacy']), 3)}

    enforcement = {}
    for pid in world.place_order:
        p = world.places[pid]
        if not p.area:
            continue
        owner = p.meta.get('sovereign')
        legit = world.factions[owner]['legitimacy'] if owner else 0.3
        enforcement[pid] = round(float(np.clip(
            float(attention[p.cells].mean()) * 0.65 + legit * 0.35, 0, 1)), 3)
        if enforcement[pid] < 0.15:
            p.tags.add('lawless')
    world.set_field('enforcement', world.place_mask_field(enforcement))
    world.artifacts['law'] = {'codes': codes, 'offences': OFFENCES,
                              'lawless_places': [p.id for p in world.places_of(tag='lawless')][:40]}
    return codes


# -------------------------------------------------------------- 44 Economy

@feature('eco.markets', 44, 'Shops, Trading, and Prices',
         'Economy & Logistics',
         consumes=('artifacts.logistics', 'fields.market_access'),
         produces=('artifacts.markets', 'fields.price_level'), part='III')
def markets(world):
    """Local prices set by distance from supply.

    Price is not a global number: it propagates along the supply network, so a
    besieged or badly connected quarter is expensive, and that shows up on the
    map as a field.
    """
    from tilegen_layers import GOODS
    supply = world.topology('logistics.supply')
    access = world.field('market_access', 0.0)
    rng = world.rng('markets')

    base_price = {g: round(float(rng.random() * 0.5 + 0.5), 3) for g in GOODS}
    prices, market_places = {}, []
    for pid in world.place_order:
        p = world.places[pid]
        if p.area < 10:
            continue
        a = float(access[p.cells].mean()) if p.area else 0.0
        local = {g: round(float(np.clip(base_price[g] * (1.6 - 0.8 * a), 0.05, 3.0)), 3)
                 for g in GOODS}
        prices[pid] = local
        if a > 0.45:
            p.tags.add('market')
            market_places.append(pid)
    level = {pid: float(np.mean(list(v.values()))) for pid, v in prices.items()}
    if level:
        mx = max(level.values()) or 1.0
        level = {k: v / mx for k, v in level.items()}
    world.set_field('price_level', world.place_mask_field(level, 0.5))
    world.artifacts['markets'] = {'base_price': base_price,
                                  'markets': market_places[:40],
                                  'sample_prices': dict(list(prices.items())[:6])}
    return prices


# ---------------------------------------------------------------- 45 Labour

@feature('soc.labour', 45, 'NPC Work and Wages',
         'Society & Institutions',
         consumes=('artifacts.markets', 'artifacts.schedule'),
         produces=('artifacts.labour',), part='III')
def labour(world):
    """Workforces, shifts and grievance.

    Grievance combines price level, wear, and seeded wages. High values add
    a workplace dispute to the generated snapshot.
    """
    ensure_population(world)
    price = world.field('price_level', 0.5)
    wear = world.field('wear', 0.3)
    rng = world.rng('labour')

    workplaces = defaultdict(list)
    for e in world.entities.values():
        workplaces[e['work']].append(e['id'])

    disputes = []
    workforce = {}
    for pid, workers in workplaces.items():
        p = world.places.get(pid)
        if p is None or len(workers) < 2:
            continue
        wage = float(rng.random() * 0.6 + 0.2)
        cost = float(price[p.cells].mean()) if p.area else 0.5
        hazard = float(wear[p.cells].mean()) if p.area else 0.3
        grievance = float(np.clip(0.5 * cost + 0.3 * hazard - 0.5 * wage + 0.2, 0, 1))
        workforce[pid] = {'workers': len(workers), 'wage': round(wage, 3),
                          'hazard': round(hazard, 3),
                          'grievance': round(grievance, 3),
                          'shift': ['dawn', 'morning', 'night'][len(workers) % 3]}
        if grievance > 0.6:
            disputes.append({'place': pid, 'kind': 'strike',
                             'grievance': round(grievance, 3),
                             'demands': ['wages', 'safety', 'hours'][len(workers) % 3]})
            p.tags.add('unrest')
            world.emit('labour.dispute', place=pid, grievance=round(grievance, 3))
    world.artifacts['labour'] = {'workforce': dict(list(workforce.items())[:60]),
                                 'disputes': disputes,
                                 'classes': CLASSES}
    return disputes


# -------------------------------------------------------------- 46 Property

@feature('soc.property', 46, 'Buildings, Ownership, and Succession',
         'Society & Institutions', consumes=('artifacts.factions',),
         produces=('artifacts.property',), part='III')
def property_rights(world):
    """Who holds what, who may enter, and who inherits.

    Access rights are separate from ownership: a steward may hold keys to a
    place they do not own, giving dungeon quests separate access and ownership rules.
    """
    ensure_population(world)
    rng = world.rng('property')
    owners, claims = {}, []
    people = list(world.entities.values())
    for pid in world.place_order:
        p = world.places[pid]
        if p.kind == 'corridor' or not people:
            continue
        holder = people[int(rng.integers(len(people)))]
        tenure = ['freehold', 'lease', 'grant', 'occupied', 'disputed'][
            int(rng.integers(5))]
        heir = people[int(rng.integers(len(people)))]
        owners[pid] = {'owner': holder['id'], 'tenure': tenure,
                       'heir': heir['id'] if heir['id'] != holder['id'] else None,
                       'access': [holder['id']]}
        if tenure == 'disputed':
            p.tags.add('disputed title')
            claims.append({'place': pid, 'claimants': [holder['id'], heir['id']]})
        # stewards hold access without ownership
        for e in people:
            if e['role'] == 'steward' and e['work'] == pid:
                owners[pid]['access'].append(e['id'])
    world.artifacts['property'] = {'titles': dict(list(owners.items())[:60]),
                                   'disputes': claims,
                                   'total_titles': len(owners)}
    return owners


# --------------------------------------- 47 Smuggler NPCs and hidden shops

@feature('eco.contraband', 47, 'Smuggler NPCs and Hidden Shops',
         'Economy & Logistics',
         consumes=('fields.enforcement', 'fields.privacy', 'artifacts.markets'),
         produces=('artifacts.contraband', 'fields.smuggling_risk'), part='III')
def contraband(world):
    """Hidden shops and smuggler routes for fictional NPC trading encounters.

    Risk combines guard attention, enforcement, shadow, and spatial privacy.
    Routes connect hidden shops through the service graph.
    """
    ensure_population(world)
    enforcement = world.field('enforcement', 0.3)
    privacy = world.field('privacy', 0.5)
    shadow = world.field('shadow', 0.5)
    attention = world.field('guard_attention', 0.2)

    concealment = np.clip(0.5 * privacy + 0.5 * shadow, 0, 1)
    risk = np.clip(attention * (0.4 + enforcement) - concealment, 0, 1).astype(np.float32)
    world.set_field('smuggling_risk', risk)

    dens, routes = [], []
    for pid in world.place_order:
        p = world.places[pid]
        if p.area < 8:
            continue
        r = float(risk[p.cells].mean())
        c = float(concealment[p.cells].mean())
        if r < 0.12 and c > 0.5:
            p.tags.add('den')
            dens.append(pid)
    service = world.topologies.get('circulation.service')
    if service and len(dens) > 1:
        for a in dens[:8]:
            dist = service.shortest_paths(a)
            for b in dens[:8]:
                if b != a and b in dist:
                    routes.append({'from': a, 'to': b, 'cost': round(dist[b], 2)})
    world.artifacts['contraband'] = {
        'goods': ['forbidden crystals', 'stolen power cells', 'forged writs',
                  'stolen relics', 'untaxed salt'],
        'dens': dens[:30], 'routes': routes[:40]}
    return dens


# ------------------------------------------------------------- 48 Identity

@feature('eco.identity', 48, 'NPC Reputation and Disguises',
         'Ecology & Inhabitants', consumes=('places',),
         produces=('artifacts.identity',), part='III')
def identity(world):
    """Identity is observer-relative: what B believes about A, not what A is.

    Recognition is a directed graph. Seeded aliases provide disguise details
    for generated investigation quests.
    """
    ensure_population(world)
    rng = world.rng('identity')
    beliefs = world.topology('society.recognition', directed=True)
    people = list(world.entities.values())
    disguises = []
    for e in people:
        if rng.random() < 0.12:
            alias = f"{GIVEN[int(rng.integers(len(GIVEN)))]} " \
                    f"{FAMILY[int(rng.integers(len(FAMILY)))]}"
            disguises.append({'entity': e['id'], 'alias': alias,
                              'quality': round(float(rng.random() * 0.5 + 0.4), 3)})
            e['alias'] = alias
    # neighbours recognise each other; strangers do not
    by_home = defaultdict(list)
    for e in people:
        by_home[e['home']].append(e['id'])
    for pid, ids in by_home.items():
        for a in ids:
            for b in ids:
                if a != b:
                    beliefs.add(a, b, round(float(rng.random() * 0.4 + 0.6), 3))
    world.artifacts['identity'] = {
        'population': len(people),
        'recognition_edges': beliefs.edge_count(),
        'disguises': disguises[:30]}
    return beliefs


# --------------------------------------- 49 Alliances and cooperation

@feature('eco.relationships', 49, 'NPC Alliances, Rivalries, and Cooperation',
         'Ecology & Inhabitants',
         consumes=('artifacts.identity',),
         produces=('artifacts.relationships', 'topology.society.alliance',
                   'topology.society.rivalry'), part='III')
def npc_relationships(world):
    """Generate unique NPC pairs with signed cooperation values.

    Positive cooperation marks alliance; negative cooperation marks rivalry.
    Separate graphs store positive strengths so graph traversal never receives
    negative edge costs. These are starting relationships for a game to use.
    """
    ensure_population(world)
    rng = world.rng('relationships')
    alliance = world.topology('society.alliance')
    rivalry = world.topology('society.rivalry')
    people = list(world.entities)
    relationships = []
    seen_pairs = set()
    for _ in range(max(3, len(people) // 3)):
        if len(people) < 2:
            break
        a, b = (people[int(i)] for i in rng.choice(len(people), size=2, replace=False))
        pair = tuple(sorted((a, b)))
        if pair in seen_pairs:
            continue
        seen_pairs.add(pair)
        kind = 'alliance' if rng.random() < 0.65 else 'rivalry'
        strength = round(float(rng.uniform(0.1, 1.0)), 3)
        cooperation = strength if kind == 'alliance' else -strength
        (alliance if kind == 'alliance' else rivalry).add(a, b, strength)
        relationships.append({'a': a, 'b': b, 'kind': kind,
                              'cooperation': cooperation})
    world.artifacts['relationships'] = {
        'count': len(relationships),
        'model': 'Starting alliances and rivalries; cooperation ranges from -1 to 1.',
        'pairs': relationships}
    return relationships


# ------------------------------------------------- 50 Ritual and Ideology

@feature('soc.ritual', 50, 'Fantasy Customs and Faction Traditions',
         'Society & Institutions',
         consumes=('places', 'artifacts.factions'),
         produces=('artifacts.ritual', 'fields.sacred_intensity'), part='III')
def ritual(world):
    """A sacred calendar and the processions that enact it.

    Sacred intensity peaks at a centre and along the procession route, so
    ideology has a geography that other systems (law, crowd, surveillance) read.
    """
    rng = world.rng('ritual')
    adjacency = world.topology('physical.adjacency')
    candidates = [p for p in world.places.values() if p.area > 8]
    if not candidates:
        world.artifacts['ritual'] = {'calendar': [], 'processions': []}
        return {}
    centre = max(candidates, key=lambda p: p.area)
    centre.tags.add('sanctum')

    dist = adjacency.shortest_paths(centre.id)
    span = max(dist.values()) if dist else 1.0
    intensity = {pid: round(max(0.0, 1.0 - d / max(span, 1e-6)), 3)
                 for pid, d in dist.items()}
    existing = world.fields.get('sacred_intensity')
    field = world.place_mask_field(intensity)
    if existing is not None:
        field = np.maximum(field, existing)
    world.set_field('sacred_intensity', field)

    stations = sorted(intensity.items(), key=lambda kv: -kv[1])[:8]
    procession = [pid for pid, _ in stations]
    calendar = []
    feasts = ['Salt Vigil', 'Ashfall', 'the Long Silence', 'Founders Rite',
              'the Reckoning']
    for i, name in enumerate(feasts):
        calendar.append({'feast': name, 'phase': i / len(feasts),
                         'observance': ['fast', 'procession', 'assembly',
                                        'lantern ceremony', 'silence'][i % 5],
                         'attendance': round(float(rng.random() * 0.5 + 0.4), 3)})
    world.artifacts['ritual'] = {'centre': centre.id, 'calendar': calendar,
                                 'procession': procession,
                                 'orthodoxy': round(float(rng.random()), 3)}
    return calendar


# ---------------------------------------------------------- 51 Surveillance

@feature('soc.surveillance', 51, 'Guard Vision and Stealth Detection',
         'Society & Institutions',
         consumes=('artifacts.perception', 'fields.guard_attention',
                   'artifacts.politics'),
         produces=('artifacts.intelligence', 'fields.observed'), part='III')
def surveillance(world):
    """Collection posts, what each can see, and where the blind spots are.

    Stealth vantage points let a character observe guards without entering
    their detection field.
    """
    rng = world.rng('surveillance')
    walk = world.passable()
    light = world.field('light', 0.5)
    observed = np.zeros((world.h, world.w), dtype=np.float32)

    candidates = sorted((p for p in world.places.values() if p.area > 6),
                        key=lambda p: -p.area)[:8]
    posts = []
    for p in candidates:
        x = int(np.clip(p.centroid[0], 0, world.w - 1))
        y = int(np.clip(p.centroid[1], 0, world.h - 1))
        vis = tw.line_of_sight(walk, (x, y),
                               radius=min(26, min(world.w, world.h) // 2))
        observed += (vis & (light > 0.08)).astype(np.float32)
        posts.append({'place': p.id, 'at': [x, y],
                      'covers': int(vis.sum()),
                      'operator': ['watch', 'informant', 'clerk'][len(posts) % 3]})
    if observed.max() > 0:
        observed /= observed.max()
    world.set_field('observed', observed)

    blind = []
    for pid in world.place_order:
        p = world.places[pid]
        if p.area >= 6 and float(observed[p.cells].mean()) < 0.05:
            p.tags.add('blind spot')
            blind.append(pid)
    world.artifacts['intelligence'] = {
        'posts': posts, 'blind_spots': blind[:30],
        'secrets_known': [c['id'] for c in
                          world.artifacts.get('politics', {}).get('conspiracies', [])
                          if rng.random() < 0.4]}
    return posts


# ------------------------------------------------------------ 52 Diplomacy

@feature('soc.diplomacy', 52, 'Faction Alliances and Negotiations',
         'Society & Institutions',
         consumes=('artifacts.factions', 'artifacts.politics'),
         produces=('artifacts.diplomacy',), part='III')
def diplomacy(world):
    """Standing between factions, and the neutral ground where it can change.

    Negotiation venues are selected from contested or unclaimed locations.
    Standings and agendas are seeded assignments.
    """
    rng = world.rng('diplomacy')
    fids = list(world.factions)
    standings, treaties = {}, []
    for i, a in enumerate(fids):
        for b in fids[i + 1:]:
            v = float(rng.random())
            state = ('war' if v < 0.15 else 'hostile' if v < 0.35 else
                     'neutral' if v < 0.65 else 'accord' if v < 0.9 else 'alliance')
            standings[f'{a}|{b}'] = {'state': state, 'trust': round(v, 3)}

    neutral = [p.id for p in world.places.values()
               if 'contested' in p.tags or p.meta.get('sovereign') is None]
    for key, st in standings.items():
        if st['state'] in ('hostile', 'war') and neutral:
            treaties.append({'parties': key.split('|'),
                             'venue': neutral[int(rng.integers(len(neutral)))],
                             'agenda': ['ceasefire', 'tariffs', 'extradition',
                                        'water rights'][int(rng.integers(4))],
                             'odds': round(float(st['trust']), 3)})
    world.artifacts['diplomacy'] = {'standings': standings,
                                    'treaty_spaces': neutral[:20],
                                    'negotiations': treaties}
    return standings


# ------------------------------------------------- 53 Violence & Escalation

ESCALATION = ['calm', 'friction', 'brawl', 'riot', 'reprisal', 'open conflict']


@feature('nar.violence', 53, 'Combat Encounters and Alert Levels',
         'Incidents & Narrative',
         consumes=('artifacts.diplomacy', 'fields.enforcement', 'artifacts.labour'),
         produces=('artifacts.violence', 'fields.threat'), part='III')
def violence(world):
    """Threat as a field and escalation as a ladder with defined rungs.

    Threat combines enforcement, contested territory, and tension. Each
    incident receives a stage label and a suggested response.
    """
    rng = world.rng('violence')
    enforcement = world.field('enforcement', 0.3)
    contested = world.field('contested', 0.0)
    tension = world.field('tension', 0.4)

    threat = np.clip(0.4 * (1 - enforcement) + 0.35 * contested +
                     0.25 * tension, 0, 1).astype(np.float32)
    world.set_field('threat', threat)

    incidents = []
    for pid in world.place_order:
        p = world.places[pid]
        if p.area < 6:
            continue
        t = float(threat[p.cells].mean())
        rung = min(len(ESCALATION) - 1, int(t * len(ESCALATION)))
        if rung >= 2:
            incidents.append({
                'place': pid, 'stage': ESCALATION[rung], 'threat': round(t, 3),
                'de_escalation': ['presence', 'arbitration', 'concession',
                                  'amnesty', 'occupation'][min(rung - 1, 4)],
                'collateral': round(float(t * rng.random()), 3)})
            p.tags.add('volatile')
            world.emit('violence.incident', place=pid, stage=ESCALATION[rung])
    world.artifacts['violence'] = {'ladder': ESCALATION, 'incidents': incidents[:40],
                                   'mean_threat': round(float(threat.mean()), 3)}
    return incidents


# ------------------------------------------------------ 54 Investigation quests

@feature('nar.investigation', 54, 'Clues and Investigation Quest Hooks',
         'Incidents & Narrative',
         consumes=('artifacts.violence', 'artifacts.identity', 'fields.observed'),
         produces=('artifacts.investigations',), part='III')
def investigation_quests(world):
    """Seed clue-linked quest hooks at combat encounter locations.

    NPC assignments are seeded choices. Clue quality describes visibility or
    surface wear in this snapshot; it does not resolve a case or name a culprit.
    """
    rng = world.rng('investigation')
    observed = world.field('observed', 0.2)
    wear = world.field('wear', 0.3)
    incidents = world.artifacts.get('violence', {}).get('incidents', [])
    disguised = {d['entity'] for d in
                 world.artifacts.get('identity', {}).get('disguises', [])}
    people = list(world.entities)

    clues, quests = [], []
    for inc in incidents[:20]:
        p = world.places.get(inc['place'])
        if p is None or not p.area:
            continue
        npc = people[int(rng.integers(len(people)))] if people else None
        clue_ids = []
        for kind, durability in (('trace', 0.3), ('witness', 0.6), ('object', 0.9)):
            cid = f'clue_{len(clues):04d}'
            clue_ids.append(cid)
            visibility = (float(observed[p.cells].mean()) if kind == 'witness'
                          else 1 - float(wear[p.cells].mean()))
            clues.append({'id': cid, 'place': p.id, 'kind': kind, 'npc': npc,
                          'quality': round(float(np.clip(visibility * durability, 0, 1)), 3)})
        quests.append({
            'id': f'inq_{len(quests):03d}', 'place': p.id, 'npc': npc,
            'objective': 'find the missing traveller' if npc else 'explore the encounter site',
            'clue_ids': clue_ids, 'uses_disguise': npc in disguised})
    world.artifacts['investigations'] = {
        'model': 'Seeded quest hooks with assigned NPCs and location-based clue quality.',
        'clues': clues, 'quests': quests}
    return quests


# --------------------------------------------------------- 55 Public Health

@feature('env.health', 55, 'Settlement Health and Environmental Hazards',
         'Environment & Infrastructure',
         consumes=('fields.crowding', 'artifacts.logistics'),
         produces=('artifacts.health', 'fields.contamination'), part='III')
def public_health(world):
    """Exposure fields and the containment policy that answers them.

    Contamination spreads through a field weighted by water and crowding.
    Hotspots list adjacent location pairs as suggested quarantine boundaries.
    """
    rng = world.rng('health')
    arr = np.array(world.tilemap.base, dtype=object).astype(str)
    crowding = world.field('crowding', 0.3)
    walk = world.passable()
    flow = np.where(np.isin(arr, ['water']), 1.0, 0.35).astype(np.float32) * walk

    seed = (crowding * walk).astype(np.float32)
    seed = np.where(seed > np.quantile(seed[walk] if walk.any() else seed, 0.9),
                    seed, 0.0)
    contamination = tw.propagate_field(seed, flow, iterations=26, decay=0.88)
    contamination = np.clip(contamination, 0, 1).astype(np.float32)
    world.set_field('contamination', contamination)

    hotspots, cordon = [], []
    for pid in world.place_order:
        p = world.places[pid]
        if p.area < 6:
            continue
        v = float(contamination[p.cells].mean())
        if v > 0.45:
            hotspots.append(pid)
            p.tags.add('contaminated')
            cordon.extend([[pid, nb] for nb in p.neighbours])
    world.artifacts['health'] = {
        'hotspots': hotspots[:30], 'quarantine_cordon': cordon[:60],
        'disease': ['salt fever', 'grey lung', 'the shakes'][
            int(rng.integers(3))],
        'mean_contamination': round(float(contamination.mean()), 3)}
    return hotspots


# ------------------------------------------------------------- 56 Scarcity

@feature('eco.scarcity', 56, 'Settlement Supplies and Resources',
         'Economy & Logistics',
         consumes=('artifacts.machines', 'artifacts.markets'),
         produces=('artifacts.scarcity', 'fields.allocation_pressure'), part='III')
def scarcity(world):
    """Estimate supply pressure and list potential machine outages.

    The dependency graph identifies machines downstream of each generator;
    utility labels describe the corresponding scenario.
    """
    rng = world.rng('scarcity')
    machines = world.artifacts.get('machines', {})
    cascade = machines.get('cascade', {})
    access = world.field('market_access', 0.0)
    crowding = world.field('crowding', 0.3)

    pressure = {}
    for pid in world.place_order:
        p = world.places[pid]
        if not p.area:
            continue
        demand = float(crowding[p.cells].mean())
        supply_ = float(access[p.cells].mean())
        pressure[pid] = round(float(np.clip(demand - supply_ + 0.3, 0, 1)), 3)
        if pressure[pid] > 0.6:
            p.tags.add('underserved')
    world.set_field('allocation_pressure', world.place_mask_field(pressure))

    outages = []
    for gen, downstream in cascade.items():
        affected = [m for m in downstream]
        outages.append({'if_lost': gen, 'machines_down': affected,
                        'utilities_lost': ['water pressure', 'vertical access'][
                            :max(1, len(affected))]})
    world.artifacts['scarcity'] = {
        'pressure_sample': dict(list(pressure.items())[:40]),
        'outages': outages,
        'rationing_threshold': round(float(rng.random() * 0.3 + 0.5), 3)}
    return pressure


# ------------------------------------------------- 57 Structural Integrity

@feature('env.structure', 57, 'Building Durability and Damage',
         'Environment & Infrastructure',
         consumes=('fields.structural_stress', 'artifacts.crises'),
         produces=('artifacts.structure', 'fields.load'), part='III')
def structural_integrity(world):
    """Load paths, damage and how a failure propagates.

    Risk combines load, wear, and stress. High-risk locations list up to four
    neighbours as possible secondary failure sites; this is a scenario overlay.
    """
    stress = world.field('structural_stress', 0.2)
    wear = world.field('wear', 0.3)
    arr = np.array(world.tilemap.base, dtype=object).astype(str)
    solid = np.isin(arr, ['wall', 'bookshelf'])

    # load accumulates downward through solid fabric
    load = np.zeros_like(stress)
    acc = np.zeros(world.w, dtype=np.float32)
    for y in range(world.h):
        acc = acc + solid[y].astype(np.float32)
        load[y] = acc
    if load.max() > 0:
        load = (load / load.max()).astype(np.float32)
    world.set_field('load', load)

    failures = []
    for pid in world.place_order:
        p = world.places[pid]
        if p.area < 4:
            continue
        risk = float(np.clip(0.5 * float(load[p.cells].mean()) +
                             0.3 * float(wear[p.cells].mean()) +
                             0.2 * float(stress[p.cells].mean()), 0, 1))
        if risk > 0.5:
            failures.append({'place': pid, 'risk': round(risk, 3),
                             'takes_with_it': sorted(p.neighbours)[:4],
                             'mode': 'collapse' if risk > 0.7 else 'partial failure'})
            p.tags.add('unsound')
    world.artifacts['structure'] = {
        'failures': failures[:40],
        'fire_paths': world.artifacts.get('crises', {}).get('fire', {}),
        'mean_load': round(float(load.mean()), 3)}
    return failures


# ---------------------------------------------------------- 58 Operations

@feature('nar.operations', 58, 'Quest Objectives and Dungeon Missions',
         'Incidents & Narrative',
         consumes=('artifacts.property', 'artifacts.intelligence',
                   'fields.smuggling_risk', 'artifacts.security'),
         produces=('artifacts.operations',), part='III')
def operations(world):
    """Generate mission outlines with objectives and guard-response references.

    Dungeon missions reference generated property, guard blind spots, and
    patrols by their game-world IDs.
    """
    rng = world.rng('operations')
    titles = world.artifacts.get('property', {}).get('titles', {})
    blind = world.artifacts.get('intelligence', {}).get('blind_spots', [])
    posts = world.artifacts.get('security', {}).get('posts', [])
    risk = world.field('smuggling_risk', 0.3)

    ops = []
    targets = [pid for pid in titles if world.places.get(pid)][:6]
    for i, target in enumerate(targets):
        p = world.places[target]
        entry = blind[i % len(blind)] if blind else target
        kind, objective = [
            ('item recovery', 'recover the lost artifact'),
            ('NPC rescue', 'rescue the stranded explorer'),
            ('disable mechanism', 'disable the barrier mechanism'),
            ('dungeon exploration', 'explore the dungeon chamber'),
        ][i % 4]
        steps = [
            {'step': 'approach', 'place': entry, 'risk': round(float(
                risk[world.places[entry].cells].mean()) if world.places.get(entry) else 0.3, 3)},
            {'step': 'reach the objective', 'place': target,
             'contact': titles[target]['owner']},
            {'step': objective, 'place': target,
             'risk': round(float(risk[p.cells].mean()), 3)},
            {'step': 'return to the entrance', 'place': entry},
        ]
        ops.append({
            'id': f'op_{i:03d}',
            'kind': kind,
            'target_place': target,
            'client': (list(world.factions)[i % len(world.factions)]
                       if world.factions else None),
            'steps': steps,
            'contingency': {'if_seen': posts[i % len(posts)] if posts else None,
                            'fallback': 'return to the entrance'},
            'payout': round(float(rng.random() * 0.7 + 0.3), 3)})
    world.artifacts['operations'] = {'contracts': ops}
    return ops


# ------------------------------------------------------------- 59 Chronicle

@feature('nar.chronicle', 59, 'World Event History',
         'Incidents & Narrative',
         consumes=('artifacts.violence', 'artifacts.investigations',
                   'artifacts.labour'),
         produces=('artifacts.chronicle',), part='III')
def chronicle(world):
    """Generate background events and quest hooks linked by faction or location."""
    nodes, edges = [], []

    def add(kind, text, causes=(), place=None):
        nid = f'evt_{len(nodes):04d}'
        nodes.append({'id': nid, 'kind': kind, 'text': text, 'place': place})
        for c in causes:
            edges.append({'from': c, 'to': nid})
        return nid

    roots = {}
    for f in world.factions.values():
        roots[f['id']] = add('founding',
                             f"{f['name'][0].upper() + f['name'][1:]} took its seat and claimed "
                             f"{len(f['territory'])} places.", place=f['seat'])
    for d in world.artifacts.get('labour', {}).get('disputes', [])[:6]:
        pid = d['place']
        owner = world.places[pid].meta.get('sovereign')
        cause = [roots[owner]] if owner in roots else []
        add('dispute', f"Workers at {world.place_name(pid)} requested changes to {d['demands']}.",
            cause, pid)
    for inc in world.artifacts.get('violence', {}).get('incidents', [])[:8]:
        prior = [n['id'] for n in nodes if n['place'] == inc['place']]
        add('violence', f"{inc['stage'].title()} at {world.place_name(inc['place'])}; "
                        f"suggested response: {inc['de_escalation']}.", prior, inc['place'])
    for quest in world.artifacts.get('investigations', {}).get('quests', [])[:6]:
        prior = [n['id'] for n in nodes if n['place'] == quest['place']]
        add('quest_hook',
            f"Clues at {world.place_name(quest['place'])} offer a quest to "
            f"{quest['objective']}.", prior, quest['place'])

    incoming = defaultdict(list)
    for e in edges:
        incoming[e['to']].append(e['from'])
    world.artifacts['chronicle'] = {
        'nodes': nodes, 'edges': edges,
        'roots': [n['id'] for n in nodes if not incoming[n['id']]],
        'prose': [n['text'] for n in nodes[:20]]}
    return nodes


# -------------------------------------------------------- 60 World Director

@feature('nar.director', 60, 'Encounter and Quest Director',
         'Incidents & Narrative',
         consumes=('artifacts.chronicle', 'fields.threat', 'artifacts.crises',
                   'artifacts.operations'),
         produces=('artifacts.director',), part='III')
def world_director(world):
    """Select seeded encounters from categories suggested by location tags.

    Each phase compares a target with a simple pacing estimate and samples
    uniformly from the matching category, falling back to all candidates.
    The estimate changes by fixed amounts; no world simulation is advanced.
    """
    rng = world.rng('director')
    threat = world.field('threat', 0.3)
    pacing_estimate = float(threat.mean())
    target_curve = [0.25, 0.4, 0.55, 0.35, 0.7, 0.5]

    beats = []
    for pid in world.place_order:
        p = world.places[pid]
        if p.area < 8:
            continue
        if 'volatile' in p.tags:
            beats.append(('escalate', pid, 'a combat encounter starts'))
        if 'contaminated' in p.tags:
            beats.append(('quarantine', pid, 'guards close a hazardous area'))
        if 'den' in p.tags:
            beats.append(('raid', pid, 'the watch moves on the den'))
        if 'unsound' in p.tags:
            beats.append(('collapse', pid, 'an unstable structure gives way'))
        if 'market' in p.tags:
            beats.append(('shortage', pid, 'supply fails to arrive'))

    schedule = []
    for i, target in enumerate(target_curve):
        want_up = target > pacing_estimate
        pool = [b for b in beats if (b[0] in ('escalate', 'collapse')) == want_up]
        if not pool:
            pool = beats
        if not pool:
            break
        pick = pool[int(rng.integers(len(pool)))]
        schedule.append({'phase': i, 'target_tension': target,
                         'beat': pick[0], 'place': pick[1],
                         'reason': pick[2]})
        pacing_estimate = pacing_estimate + (0.1 if want_up else -0.08)
        world.emit('director.scheduled', beat=pick[0], place=pick[1])

    world.artifacts['director'] = {
        'initial_threat': round(float(threat.mean()), 3),
        'selection': 'seeded choice within encounter categories',
        'target_curve': target_curve,
        'candidate_beats': len(beats),
        'schedule': schedule,
        'chronicle_size': len(world.artifacts.get('chronicle', {}).get('nodes', []))}
    return schedule
