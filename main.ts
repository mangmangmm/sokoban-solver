// sokoban-solver: a push-optimal Sokoban solver.
//
// One program answers a set of 5 levels. Every level is a hallway with
// rooms off it, and the rooms are independent: a box never leaves its
// room, so the level's optimal push count is the sum of its rooms'
// optima. The solver splits a level into rooms, solves each room with
// a breadth-first search over pushes (the first solution a BFS meets
// is push-optimal), and joins the rooms' move strings.
//
// A search state is the boxes' squares plus the region the player can
// reach (named by its lowest square: two regions with the same lowest
// square are the same region, since regions never overlap). A state is
// pruned when a box sits on a square from which no goal can ever be
// reached, or inside a frozen 2x2 block of walls and boxes. The search
// keeps at most MAX_STATES states per room, a work budget, never the
// clock, so two runs answer byte for byte alike.
//
// Local run: mkdir -p out && MAKEFUN_INPUT_DIR=practice/input MAKEFUN_OUTPUT_DIR=out node main.ts

import { mkdirSync, readdirSync, readFileSync, writeFileSync } from 'node:fs';
import { join } from 'node:path';

const INPUT_DIR = process.env.MAKEFUN_INPUT_DIR ?? '/input';
const OUTPUT_DIR = process.env.MAKEFUN_OUTPUT_DIR ?? '/output';
/** The most states one room search keeps: a work budget, never the clock. */
const MAX_STATES = 250_000;

interface Step {
  dx: number;
  dy: number;
  walk: string;
  push: string;
}
const STEPS: Step[] = [
  { dx: 0, dy: -1, walk: 'u', push: 'U' },
  { dx: 0, dy: 1, walk: 'd', push: 'D' },
  { dx: -1, dy: 0, walk: 'l', push: 'L' },
  { dx: 1, dy: 0, walk: 'r', push: 'R' },
];

interface Board {
  w: number;
  h: number;
  /** 1 is a wall. */
  wall: Uint8Array;
  goals: number[];
  boxes: number[];
  player: number;
}

interface Answer {
  pushes: number;
  moves: string;
}

/** Parse XSB text into a board. Short rows and the outer frame count as walls. */
function parseLevel(text: string): Board {
  const rows = text.split('\n').filter((r) => r.length > 0);
  const h = rows.length;
  const w = Math.max(...rows.map((r) => r.length));
  const wall = new Uint8Array(w * h).fill(1);
  const goals: number[] = [];
  const boxes: number[] = [];
  let player = -1;
  rows.forEach((row, y) => {
    for (let x = 0; x < row.length; x++) {
      const ch = row[x];
      const i = y * w + x;
      if (ch === '#') continue;
      wall[i] = 0;
      if (ch === '.' || ch === '*' || ch === '+') goals.push(i);
      if (ch === '$' || ch === '*') boxes.push(i);
      if (ch === '@' || ch === '+') player = i;
    }
  });
  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) {
      if (x === 0 || y === 0 || x === w - 1 || y === h - 1) wall[y * w + x] = 1;
    }
  }
  return { w, h, wall, goals, boxes: boxes.sort((a, b) => a - b), player };
}

/**
 * The squares a box can be pushed from onto some goal: pull back from
 * the goals. A box on square y can be pulled onto x when the player can
 * stand on the far side, y - (x - y), so a square is live when a chain
 * of such pulls reaches a goal. A square no chain reaches is dead: a
 * box there can never reach a goal, whatever the player does.
 */
function liveSquares(b: Board): Uint8Array {
  const live = new Uint8Array(b.w * b.h);
  const queue: number[] = [];
  for (const g of b.goals) {
    live[g] = 1;
    queue.push(g);
  }
  for (let qi = 0; qi < queue.length; qi++) {
    const x = queue[qi];
    for (const s of STEPS) {
      const d = s.dy * b.w + s.dx;
      const y = x - d;
      if (!b.wall[y] && !b.wall[y - d] && !live[y]) {
        live[y] = 1;
        queue.push(y);
      }
    }
  }
  return live;
}

/**
 * Flood the squares the player reaches from `from` without crossing a
 * wall or a box, stamping them in `seen`; answer the lowest reached
 * square, which names the whole region.
 */
function floodRegion(
  b: Board,
  occupied: Uint8Array,
  from: number,
  seen: Int32Array,
  stamp: number,
  stack: Int32Array,
): number {
  let low = from;
  let n = 0;
  stack[n++] = from;
  seen[from] = stamp;
  while (n > 0) {
    const p = stack[--n];
    if (p < low) low = p;
    for (const s of STEPS) {
      const q = p + s.dy * b.w + s.dx;
      if (!b.wall[q] && !occupied[q] && seen[q] !== stamp) {
        seen[q] = stamp;
        stack[n++] = q;
      }
    }
  }
  return low;
}

/**
 * Whether the box just pushed onto `t` is stuck in a frozen 2x2 block:
 * four squares of walls and boxes, one of the boxes off its goal.
 */
function frozenBlock(b: Board, occupied: Uint8Array, goal: Uint8Array, t: number): boolean {
  const w = b.w;
  for (const a of [t, t - 1, t - w, t - w - 1]) {
    const cells = [a, a + 1, a + w, a + w + 1];
    const filled = cells.every((c) => b.wall[c] === 1 || occupied[c] === 1);
    if (!filled) continue;
    if (cells.some((c) => occupied[c] === 1 && goal[c] === 0)) return true;
  }
  return false;
}

/** The shortest walk from `from` to `to` around walls and boxes, in lower-case letters. */
function walkTo(b: Board, occupied: Uint8Array, from: number, to: number): string {
  if (from === to) return '';
  const prev = new Int32Array(b.w * b.h).fill(-1);
  const how: string[] = new Array(b.w * b.h);
  const queue = [from];
  prev[from] = from;
  for (let qi = 0; qi < queue.length; qi++) {
    const p = queue[qi];
    if (p === to) break;
    for (const s of STEPS) {
      const q = p + s.dy * b.w + s.dx;
      if (b.wall[q] || occupied[q] || prev[q] !== -1) continue;
      prev[q] = p;
      how[q] = s.walk;
      queue.push(q);
    }
  }
  if (prev[to] === -1) throw new Error('walkTo: push square out of reach');
  let moves = '';
  for (let p = to; p !== from; p = prev[p]) moves = how[p] + moves;
  return moves;
}

/** The square the player stands on after `moves` from `from`. */
function endOf(w: number, from: number, moves: string): number {
  let p = from;
  for (const ch of moves.toLowerCase()) {
    p += ch === 'u' ? -w : ch === 'd' ? w : ch === 'l' ? -1 : 1;
  }
  return p;
}

/**
 * Push-optimal solution of one board: BFS over pushes, so the first
 * solved state met is optimal. A state is the boxes plus the player's
 * region; pushing is the only action, and the walk between pushes is
 * filled in when the path is rebuilt. Null when the search gives up
 * (past MAX_STATES states) or the board has no solution.
 */
function solvePushOptimal(b: Board, maxStates: number): Answer | null {
  const size = b.w * b.h;
  const goal = new Uint8Array(size);
  for (const g of b.goals) goal[g] = 1;
  const live = liveSquares(b);
  for (const box of b.boxes) if (!live[box]) return null;

  const occupied = new Uint8Array(size);
  const seen = new Int32Array(size);
  const mark = new Int32Array(size);
  const stack = new Int32Array(size);
  let stamp = 0;

  const solved = (boxes: number[]): boolean => boxes.every((x) => goal[x] === 1);
  const keyOf = (boxes: number[], low: number): string => `${low}|${boxes.join(',')}`;

  const stateBoxes: number[][] = [b.boxes.slice()];
  const statePlayer: number[] = [b.player];
  const parent: number[] = [-1];
  const index = new Map<string, number>();

  for (const box of b.boxes) occupied[box] = 1;
  const startLow = floodRegion(b, occupied, b.player, seen, ++stamp, stack);
  for (const box of b.boxes) occupied[box] = 0;
  index.set(keyOf(b.boxes, startLow), 0);

  let found = solved(b.boxes) ? 0 : -1;
  for (let head = 0; found < 0 && head < stateBoxes.length; head++) {
    const boxes = stateBoxes[head];
    for (const box of boxes) occupied[box] = 1;
    const here = ++stamp;
    floodRegion(b, occupied, statePlayer[head], seen, here, stack);
    for (let bi = 0; bi < boxes.length && found < 0; bi++) {
      const box = boxes[bi];
      for (const s of STEPS) {
        const d = s.dy * b.w + s.dx;
        const t = box + d;
        if (seen[box - d] !== here) continue;
        if (b.wall[t] || occupied[t] || !live[t]) continue;
        occupied[box] = 0;
        occupied[t] = 1;
        const frozen = frozenBlock(b, occupied, goal, t);
        let key = '';
        let next: number[] | null = null;
        if (!frozen) {
          next = boxes.slice();
          next[bi] = t;
          next.sort((x, y) => x - y);
          const low = floodRegion(b, occupied, box, mark, ++stamp, stack);
          key = keyOf(next, low);
        }
        occupied[t] = 0;
        occupied[box] = 1;
        if (frozen || index.has(key)) continue;
        if (stateBoxes.length >= maxStates) {
          for (const x of boxes) occupied[x] = 0;
          return null;
        }
        index.set(key, stateBoxes.length);
        stateBoxes.push(next!);
        statePlayer.push(box);
        parent.push(head);
        if (solved(next!)) {
          found = stateBoxes.length - 1;
          break;
        }
      }
    }
    for (const x of boxes) occupied[x] = 0;
  }
  if (found < 0) return null;

  const path: number[] = [];
  for (let s = found; s > 0; s = parent[s]) path.push(s);
  path.reverse();
  let moves = '';
  let player = b.player;
  let boxes = b.boxes.slice();
  try {
    for (const s of path) {
      const nextBoxes = stateBoxes[s];
      const from = boxes.find((x) => !nextBoxes.includes(x))!;
      const to = nextBoxes.find((x) => !boxes.includes(x))!;
      const d = to - from;
      const step = STEPS.find((x) => x.dy * b.w + x.dx === d)!;
      for (const x of boxes) occupied[x] = 1;
      moves += walkTo(b, occupied, player, from - d);
      for (const x of boxes) occupied[x] = 0;
      moves += step.push;
      player = from;
      boxes = nextBoxes;
    }
  } catch {
    // A push square out of reach means the path is unusable: give up
    // this board rather than crash the run, so the other levels still
    // get their answers.
    return null;
  }
  return { pushes: path.length, moves };
}

interface Room {
  squares: number[];
  boxes: number[];
  goals: number[];
  done?: boolean;
}

/**
 * A level's rooms: cut the dead squares of the player's row (the
 * hallway: a box there reaches no goal) out of the floor, and each
 * part left that holds a box is a room. Null when some part holds more
 * or fewer boxes than goals, a goal falls in the cut, or a box starts
 * in the hallway: then the level is no set of rooms.
 */
function roomsOf(b: Board): Room[] | null {
  const live = liveSquares(b);
  const row = Math.floor(b.player / b.w);
  const cut = new Uint8Array(b.w * b.h);
  for (let i = row * b.w; i < (row + 1) * b.w; i++) {
    if (!b.wall[i] && !live[i]) cut[i] = 1;
  }
  if (b.boxes.some((x) => cut[x])) return null;
  const part = new Int32Array(b.w * b.h).fill(-1);
  const rooms: Room[] = [];
  for (let s = 0; s < b.w * b.h; s++) {
    if (b.wall[s] || cut[s] || part[s] >= 0) continue;
    const squares = [s];
    part[s] = rooms.length;
    for (let i = 0; i < squares.length; i++) {
      for (const st of STEPS) {
        const q = squares[i] + st.dy * b.w + st.dx;
        if (!b.wall[q] && !cut[q] && part[q] < 0) {
          part[q] = rooms.length;
          squares.push(q);
        }
      }
    }
    rooms.push({ squares, boxes: [], goals: [] });
  }
  for (const box of b.boxes) rooms[part[box]].boxes.push(box);
  for (const g of b.goals) if (part[g] >= 0) rooms[part[g]].goals.push(g);
  if (b.goals.some((g) => part[g] < 0)) return null;
  if (rooms.some((r) => r.boxes.length !== r.goals.length)) return null;
  return rooms.filter((r) => r.boxes.length > 0);
}

/**
 * A level solved room by room: each room's push-optimal answer in
 * turn, the other rooms' boxes standing as walls (on their goals once
 * their room is done), the player walking on from where the last room
 * left it. A box never leaves its room, so the rooms' optima add up
 * to the level's. Null when a room's search gives up, or the level is
 * no set of rooms and the search over the whole level gives up.
 */
function solveLevel(b: Board): Answer | null {
  const rooms = roomsOf(b);
  if (rooms === null) return solvePushOptimal(b, MAX_STATES);
  let player = b.player;
  let pushes = 0;
  let moves = '';
  for (const room of rooms) {
    const wall = b.wall.slice();
    for (const other of rooms) {
      if (other === room) continue;
      for (const x of other.done ? other.goals : other.boxes) wall[x] = 1;
    }
    const sub: Board = {
      w: b.w,
      h: b.h,
      wall,
      goals: room.goals.slice().sort((a, b2) => a - b2),
      boxes: room.boxes.slice().sort((a, b2) => a - b2),
      player,
    };
    const solved = solvePushOptimal(sub, MAX_STATES);
    if (solved === null) return null;
    pushes += solved.pushes;
    moves += solved.moves;
    player = endOf(b.w, player, solved.moves);
    room.done = true;
  }
  return { pushes, moves };
}

for (const id of readdirSync(INPUT_DIR).sort()) {
  let moves = '';
  try {
    const answer = solveLevel(parseLevel(readFileSync(join(INPUT_DIR, id, 'level.txt'), 'utf8')));
    moves = answer === null ? '' : answer.moves;
  } catch {
    // One level failing must not take the other four down with it.
    moves = '';
  }
  mkdirSync(join(OUTPUT_DIR, id), { recursive: true });
  writeFileSync(join(OUTPUT_DIR, id, 'solution.txt'), `${moves}\n`);
}
