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
        n: usize, time: usize,
        mut t: [Chars; n]
    }

    let mut vacancy = (0, 0);
    for i in 0..n {
        for j in 0..n {
            if t[i][j] == '0' {
                vacancy = (i as i64, j as i64);
                break;
            }
        }
    }

    let mut answer = String::new();
    let movement = ['R', 'D', 'L', 'U'];
    //let mut score = 0;
    let d = [(0, 1), (1, 0), (0, -1), (-1, 0)];

    let mut count = 0;
    while count < time {
        let time = Instant::now();
        if time.duration_since(start).as_secs_f64() > 1.9 {
            break;
        }

        for (i, &dij) in d.iter().enumerate() {
            if solve(n, &mut vacancy, dij, &mut t) {
                answer.push(movement[i]);
                count += 1;
                break;
            }
        }
    }

    println!("{}", answer);
}

fn get_score(n: usize, vacancy: &mut (i64, i64), d: (i64, i64), t: &mut Vec<Vec<char>>) -> usize {
    rand::thread_rng().gen_range(0, 101)
}

fn solve(n: usize, vacancy: &mut (i64, i64), d: (i64, i64), t: &mut Vec<Vec<char>>) -> bool {
    let slide = (vacancy.0 + d.0, vacancy.1 + d.1);
    if !((0 <= slide.0 && slide.0 < n as i64) && (0 <= slide.1 && slide.1 < n as i64)) {
        return false;
    }

    if get_score(n, vacancy, d, t) >= 50 {
        return false;
    }
    // タイルの入れ替え
    t[vacancy.0 as usize][vacancy.1 as usize] = t[slide.0 as usize][slide.1 as usize];
    t[slide.0 as usize][slide.1 as usize] = '0';
    vacancy.0 += d.0;
    vacancy.1 += d.1;

    true
}
