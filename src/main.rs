#[allow(unused_imports)]
use proconio::{
    fastout, input,
    marker::{Bytes, Chars},
};

use rand::Rng;
use std::time::Instant;

#[fastout]
fn main() {
    let start = Instant::now();
    input! {
        n: usize, T: usize,
        mut tile: [Chars; n]
    }

    let mut TILE = vec![vec![0; n]; n];
    for i in 0..n {
        for j in 0..n {
            let tmp = tile[i][j];
            let tmp_num = tmp as i64 - '0' as i64;
            if 0 <= tmp_num && tmp_num < 10 {
                TILE[i][j] = tmp_num as usize;
            } else {
                let tmp_num = tmp as i64 - 'a' as i64;
                TILE[i][j] = 10 + tmp_num as usize;
            }
        }
    }

    let mut vacancy = (0, 0);
    for i in 0..n {
        for j in 0..n {
            if TILE[i][j] == 0 {
                vacancy = (i as i64, j as i64);
                break;
            }
        }
    }

    let tmp_TILE = TILE.clone();
    let TILE_Input = Input {
        n: n,
        T: T,
        tiles: tmp_TILE,
    };
    let sim = Sim::new(&TILE_Input);
    let (mut score, _, _) = sim.compute_score(&TILE_Input);

    let mut result = (String::new(), score);
    let movement = ['R', 'D', 'L', 'U'];
    let d = [(0, 1), (1, 0), (0, -1), (-1, 0)];

    loop {
        let time = Instant::now();
        if time.duration_since(start).as_secs_f64() > 2.9 {
            break;
        }

        let mut answer = String::new();
        let mut count = 0;
        let mut tmp_tile = TILE.clone();
        let mut tmp_vacancy = vacancy.clone();
        while count < T {
            let temp = rand::thread_rng().gen_range(0, 101);
            let next_point = rand::thread_rng().gen_range(0, 4);
            let dij = d[next_point];

            if !solve(n, &mut tmp_vacancy, dij, &mut tmp_tile, T, temp, &mut score) {
                continue;
            }
            answer.push(movement[next_point]);
            count += 1;

            if result.1 < score {
                result = (answer.clone(), score);
                //println!("{:#?}", result);
            }
        }
    }

    println!("{}", result.0);
}

fn solve(
    n: usize,
    vacancy: &mut (i64, i64),
    d: (i64, i64),
    tile: &mut Vec<Vec<usize>>,
    T: usize,
    temp: usize,
    score: &mut i64,
) -> bool {
    let slide = (vacancy.0 + d.0, vacancy.1 + d.1);
    if !((0 <= slide.0 && slide.0 < n as i64) && (0 <= slide.1 && slide.1 < n as i64)) {
        return false;
    }

    swap_tile(tile, slide, *vacancy);

    //tile[vacancy.0 as usize][vacancy.1 as usize] = tile[slide.0 as usize][slide.1 as usize];
    //tile[slide.0 as usize][slide.1 as usize] = 0;

    let TILE = Input {
        n: n,
        T: T,
        tiles: tile.clone(),
    };

    let sim = Sim::new(&TILE);
    let (next_score, _, _) = sim.compute_score(&TILE);
    if next_score < *score {
        if temp < 90 {
            swap_tile(tile, *vacancy, slide);
            //tile[slide.0 as usize][slide.1 as usize] = tile[vacancy.0 as usize][vacancy.1 as usize];
            //tile[vacancy.0 as usize][vacancy.1 as usize] = 0;
            return false;
        }
    }
    vacancy.0 += d.0;
    vacancy.1 += d.1;
    *score = next_score;

    true
}

fn swap_tile(tile: &mut Vec<Vec<usize>>, slide: (i64, i64), vacancy: (i64, i64)) {
    tile[vacancy.0 as usize][vacancy.1 as usize] = tile[slide.0 as usize][slide.1 as usize];
    tile[slide.0 as usize][slide.1 as usize] = 0;
}

pub trait SetMinMax {
    fn setmin(&mut self, v: Self) -> bool;
    fn setmax(&mut self, v: Self) -> bool;
}
impl<T> SetMinMax for T
where
    T: PartialOrd,
{
    fn setmin(&mut self, v: T) -> bool {
        *self > v && {
            *self = v;
            true
        }
    }
    fn setmax(&mut self, v: T) -> bool {
        *self < v && {
            *self = v;
            true
        }
    }
}

#[macro_export]
macro_rules! mat {
	($($e:expr),*) => { Vec::from(vec![$($e),*]) };
	($($e:expr,)*) => { Vec::from(vec![$($e),*]) };
	($e:expr; $d:expr) => { Vec::from(vec![$e; $d]) };
	($e:expr; $d:expr $(; $ds:expr)+) => { Vec::from(vec![mat![$e $(; $ds)*]; $d]) };
}

use std::cell::Cell;

#[derive(Clone, Debug)]
pub struct UnionFind {
    /// size / parent
    ps: Vec<Cell<usize>>,
    pub is_root: Vec<bool>,
}

impl UnionFind {
    pub fn new(n: usize) -> UnionFind {
        UnionFind {
            ps: vec![Cell::new(1); n],
            is_root: vec![true; n],
        }
    }
    pub fn find(&self, x: usize) -> usize {
        if self.is_root[x] {
            x
        } else {
            let p = self.find(self.ps[x].get());
            self.ps[x].set(p);
            p
        }
    }
    pub fn unite(&mut self, x: usize, y: usize) {
        let mut x = self.find(x);
        let mut y = self.find(y);
        if x == y {
            return;
        }
        if self.ps[x].get() < self.ps[y].get() {
            ::std::mem::swap(&mut x, &mut y);
        }
        *self.ps[x].get_mut() += self.ps[y].get();
        self.ps[y].set(x);
        self.is_root[y] = false;
    }
    pub fn same(&self, x: usize, y: usize) -> bool {
        self.find(x) == self.find(y)
    }
    pub fn size(&self, x: usize) -> usize {
        self.ps[self.find(x)].get()
    }
}

pub type Output = Vec<char>;

pub const DIJ: [(usize, usize); 4] = [(0, !0), (!0, 0), (0, 1), (1, 0)];
pub const DIR: [char; 4] = ['L', 'U', 'R', 'D'];

#[derive(Clone, Debug)]
pub struct Input {
    pub n: usize,
    pub T: usize,
    pub tiles: Vec<Vec<usize>>,
}

impl std::fmt::Display for Input {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "{} {}", self.n, self.T)?;
        for i in 0..self.n {
            for j in 0..self.n {
                write!(f, "{:0x}", self.tiles[i][j])?;
            }
            writeln!(f)?;
        }
        Ok(())
    }
}

pub fn parse_input(f: &str) -> Input {
    let f = proconio::source::once::OnceSource::from(f);
    input! {
        from f,
        n: usize,
        T: usize,
        tiles: [Chars; n]
    }
    let tiles = tiles
        .iter()
        .map(|ts| {
            ts.iter()
                .map(|&c| usize::from_str_radix(&c.to_string(), 16).unwrap())
                .collect()
        })
        .collect();
    Input { n, T, tiles }
}

pub fn parse_output(_input: &Input, f: &str) -> Result<Output, String> {
    Ok(f.trim().chars().collect())
}

pub struct Sim {
    n: usize,
    T: usize,
    from: Vec<Vec<(usize, usize)>>,
    turn: usize,
    i: usize,
    j: usize,
}

impl Sim {
    pub fn new(input: &Input) -> Self {
        let mut i = !0;
        let mut j = !0;
        let mut from = mat![(0, 0); input.n; input.n];
        for x in 0..input.n {
            for y in 0..input.n {
                if input.tiles[x][y] == 0 {
                    i = x;
                    j = y;
                }
                from[x][y] = (x, y);
            }
        }
        Sim {
            n: input.n,
            T: input.T,
            from,
            turn: 0,
            i,
            j,
        }
    }
    pub fn apply(&mut self, c: char) -> Result<(), String> {
        if let Some(d) = DIR.iter().position(|&d| d == c) {
            let i2 = self.i + DIJ[d].0;
            let j2 = self.j + DIJ[d].1;
            if i2 >= self.n || j2 >= self.n {
                Err(format!("illegal move: {} (turn {})", c, self.turn))
            } else {
                let f1 = self.from[self.i][self.j];
                let f2 = self.from[i2][j2];
                self.from[i2][j2] = f1;
                self.from[self.i][self.j] = f2;
                self.i = i2;
                self.j = j2;
                self.turn += 1;
                Ok(())
            }
        } else {
            Err(format!("illegal move: {} (turn {})", c, self.turn))
        }
    }
    pub fn compute_score(&self, input: &Input) -> (i64, String, Vec<Vec<bool>>) {
        let mut uf = UnionFind::new(self.n * self.n);
        let mut tree = vec![true; self.n * self.n];
        let mut tiles = mat![0; self.n; self.n];
        for i in 0..self.n {
            for j in 0..self.n {
                tiles[i][j] = input.tiles[self.from[i][j].0][self.from[i][j].1];
            }
        }
        for i in 0..self.n {
            for j in 0..self.n {
                if i + 1 < self.n && tiles[i][j] & 8 != 0 && tiles[i + 1][j] & 2 != 0 {
                    let a = uf.find(i * self.n + j);
                    let b = uf.find((i + 1) * self.n + j);
                    if a == b {
                        tree[a] = false;
                    } else {
                        let t = tree[a] && tree[b];
                        uf.unite(a, b);
                        tree[uf.find(a)] = t;
                    }
                }
                if j + 1 < self.n && tiles[i][j] & 4 != 0 && tiles[i][j + 1] & 1 != 0 {
                    let a = uf.find(i * self.n + j);
                    let b = uf.find(i * self.n + j + 1);
                    if a == b {
                        tree[a] = false;
                    } else {
                        let t = tree[a] && tree[b];
                        uf.unite(a, b);
                        tree[uf.find(a)] = t;
                    }
                }
            }
        }
        let mut max_tree = !0;
        for i in 0..self.n {
            for j in 0..self.n {
                if tiles[i][j] != 0 && tree[uf.find(i * self.n + j)] {
                    if max_tree == !0 || uf.size(max_tree) < uf.size(i * self.n + j) {
                        max_tree = i * self.n + j;
                    }
                }
            }
        }
        let mut bs = mat![false; self.n; self.n];
        if max_tree != !0 {
            for i in 0..self.n {
                for j in 0..self.n {
                    bs[i][j] = uf.same(max_tree, i * self.n + j);
                }
            }
        }
        if self.turn > self.T {
            return (0, format!("too many moves"), bs);
        }
        let size = if max_tree == !0 { 0 } else { uf.size(max_tree) };
        let score = if size == self.n * self.n - 1 {
            (500000.0 * (1.0 + (self.T - self.turn) as f64 / self.T as f64)).round()
        } else {
            (500000.0 * size as f64 / (self.n * self.n - 1) as f64).round()
        } as i64;
        (score, String::new(), bs)
    }
}
