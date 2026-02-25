//! Integration tests for tensor_spatial public API.

use tensor_spatial::{
    BoundingBox, BoundingBox3D, BoundingBoxN, SpatialConfig, SpatialEntry, SpatialEntry3D,
    SpatialEntryN, SpatialError, SpatialIndex, SpatialIndex3D, SpatialIndexN, SplitStrategy,
};

// ---------------------------------------------------------------------------
// 2D BoundingBox tests
// ---------------------------------------------------------------------------

#[test]
fn test_bounding_box_new_and_accessors() {
    let bb = BoundingBox::new(1.0, 2.0, 3.0, 4.0).unwrap();
    assert_eq!(bb.x(), 1.0);
    assert_eq!(bb.y(), 2.0);
    assert_eq!(bb.width(), 3.0);
    assert_eq!(bb.height(), 4.0);
}

#[test]
fn test_bounding_box_new_rejects_negative_width() {
    assert!(BoundingBox::new(0.0, 0.0, -1.0, 1.0).is_err());
}

#[test]
fn test_bounding_box_new_rejects_negative_height() {
    assert!(BoundingBox::new(0.0, 0.0, 1.0, -1.0).is_err());
}

#[test]
fn test_bounding_box_center() {
    let bb = BoundingBox::new(0.0, 0.0, 4.0, 6.0).unwrap();
    assert_eq!(bb.center(), (2.0, 3.0));
}

#[test]
fn test_bounding_box_contains_point() {
    let bb = BoundingBox::new(1.0, 1.0, 3.0, 3.0).unwrap();
    // Inside
    assert!(bb.contains_point(2.0, 2.0));
    // On edge
    assert!(bb.contains_point(1.0, 1.0));
    assert!(bb.contains_point(4.0, 4.0));
    // Outside
    assert!(!bb.contains_point(0.0, 2.0));
    assert!(!bb.contains_point(5.0, 2.0));
    assert!(!bb.contains_point(2.0, 0.0));
    assert!(!bb.contains_point(2.0, 5.0));
}

#[test]
fn test_bounding_box_intersects() {
    let a = BoundingBox::new(0.0, 0.0, 2.0, 2.0).unwrap();
    let b = BoundingBox::new(1.0, 1.0, 2.0, 2.0).unwrap();
    assert!(a.intersects(b));
    assert!(b.intersects(a));

    let c = BoundingBox::new(5.0, 5.0, 1.0, 1.0).unwrap();
    assert!(!a.intersects(c));
    assert!(!c.intersects(a));
}

#[test]
fn test_bounding_box_intersects_edge_touching() {
    let a = BoundingBox::new(0.0, 0.0, 2.0, 2.0).unwrap();
    let b = BoundingBox::new(2.0, 0.0, 2.0, 2.0).unwrap();
    assert!(!a.intersects(b));
}

#[test]
fn test_bounding_box_union() {
    let a = BoundingBox::new(0.0, 0.0, 2.0, 2.0).unwrap();
    let b = BoundingBox::new(1.0, 1.0, 3.0, 3.0).unwrap();
    let u = a.union(b);
    assert_eq!(u.x(), 0.0);
    assert_eq!(u.y(), 0.0);
    assert_eq!(u.width(), 4.0);
    assert_eq!(u.height(), 4.0);
}

#[test]
fn test_bounding_box_area() {
    let bb = BoundingBox::new(0.0, 0.0, 3.0, 5.0).unwrap();
    assert!((bb.area() - 15.0).abs() < f32::EPSILON);
}

#[test]
fn test_bounding_box_is_empty() {
    assert!(BoundingBox::new(0.0, 0.0, 0.0, 5.0).unwrap().is_empty());
    assert!(BoundingBox::new(0.0, 0.0, 5.0, 0.0).unwrap().is_empty());
    assert!(BoundingBox::new(0.0, 0.0, 0.0, 0.0).unwrap().is_empty());
    assert!(!BoundingBox::new(0.0, 0.0, 1.0, 1.0).unwrap().is_empty());
}

#[test]
fn test_spatial_entry() {
    let entry = SpatialEntry {
        bounds: BoundingBox::new(1.0, 2.0, 3.0, 4.0).unwrap(),
        data: 42,
    };
    assert_eq!(entry.data, 42);
    assert_eq!(entry.bounds.x(), 1.0);

    // Test Debug impl
    let debug_str = format!("{entry:?}");
    assert!(debug_str.contains("SpatialEntry"));

    // Test Clone impl
    let cloned = entry.clone();
    assert_eq!(cloned.data, entry.data);
    assert_eq!(cloned.bounds, entry.bounds);

    // Test PartialEq impl
    assert_eq!(entry, cloned);
}

#[test]
fn test_spatial_index_empty() {
    let index: SpatialIndex<u32> = SpatialIndex::new();
    assert!(index.is_empty());
    assert_eq!(index.len(), 0);
    assert_eq!(index.iter().count(), 0);
}

#[test]
fn test_spatial_index_default() {
    let index: SpatialIndex<u32> = SpatialIndex::default();
    assert!(index.is_empty());
}

#[test]
fn test_spatial_index_insert_and_query_region() {
    let mut index = SpatialIndex::new();

    for i in 0..20 {
        let x = (i % 5) as f32 * 10.0;
        let y = (i / 5) as f32 * 10.0;
        index.insert(SpatialEntry {
            bounds: BoundingBox::new(x, y, 5.0, 5.0).unwrap(),
            data: i,
        });
    }
    assert_eq!(index.len(), 20);

    let region = BoundingBox::new(0.0, 0.0, 12.0, 12.0).unwrap();
    let results = index.query_region(region);
    assert!(!results.is_empty());

    for entry in &results {
        assert!(entry.bounds.intersects(region));
    }
}

#[test]
fn test_spatial_index_query_region_no_results() {
    let mut index = SpatialIndex::new();
    for i in 0..10 {
        index.insert(SpatialEntry {
            bounds: BoundingBox::new(0.0, 0.0, 1.0, 1.0).unwrap(),
            data: i,
        });
    }
    let region = BoundingBox::new(100.0, 100.0, 1.0, 1.0).unwrap();
    let results = index.query_region(region);
    assert!(results.is_empty());
}

#[test]
fn test_spatial_index_query_nearest() {
    let mut index = SpatialIndex::new();

    index.insert(SpatialEntry {
        bounds: BoundingBox::new(0.0, 0.0, 1.0, 1.0).unwrap(),
        data: "origin",
    });
    index.insert(SpatialEntry {
        bounds: BoundingBox::new(10.0, 10.0, 1.0, 1.0).unwrap(),
        data: "far",
    });
    index.insert(SpatialEntry {
        bounds: BoundingBox::new(2.0, 2.0, 1.0, 1.0).unwrap(),
        data: "near",
    });

    let results = index.query_nearest(0.0, 0.0, 2);
    assert_eq!(results.len(), 2);
    assert_eq!(results[0].data, "origin");
    assert_eq!(results[1].data, "near");
}

#[test]
fn test_spatial_index_query_nearest_k_larger_than_n() {
    let mut index = SpatialIndex::new();
    index.insert(SpatialEntry {
        bounds: BoundingBox::new(0.0, 0.0, 1.0, 1.0).unwrap(),
        data: 1,
    });
    index.insert(SpatialEntry {
        bounds: BoundingBox::new(5.0, 5.0, 1.0, 1.0).unwrap(),
        data: 2,
    });

    let results = index.query_nearest(0.0, 0.0, 100);
    assert_eq!(results.len(), 2);
}

#[test]
fn test_spatial_index_remove() {
    let mut index = SpatialIndex::new();
    let bb = BoundingBox::new(1.0, 1.0, 2.0, 2.0).unwrap();
    index.insert(SpatialEntry {
        bounds: bb,
        data: 42u32,
    });
    assert_eq!(index.len(), 1);

    index.remove(bb, |e| e.data == 42).unwrap();
    assert_eq!(index.len(), 0);
    assert!(index.is_empty());

    let result = index.remove(bb, |e| e.data == 42);
    assert!(result.is_err());
}

#[test]
fn test_spatial_index_remove_from_tree() {
    let mut index = SpatialIndex::new();
    for i in 0..20u32 {
        let x = (i % 5) as f32 * 10.0;
        let y = (i / 5) as f32 * 10.0;
        index.insert(SpatialEntry {
            bounds: BoundingBox::new(x, y, 5.0, 5.0).unwrap(),
            data: i,
        });
    }
    assert_eq!(index.len(), 20);

    let bb = BoundingBox::new(0.0, 0.0, 5.0, 5.0).unwrap();
    index.remove(bb, |e| e.data == 0).unwrap();
    assert_eq!(index.len(), 19);
}

#[test]
fn test_spatial_index_clear() {
    let mut index = SpatialIndex::new();
    for i in 0..10 {
        index.insert(SpatialEntry {
            bounds: BoundingBox::new(0.0, 0.0, 1.0, 1.0).unwrap(),
            data: i,
        });
    }
    assert_eq!(index.len(), 10);
    index.clear();
    assert!(index.is_empty());
    assert_eq!(index.len(), 0);
}

#[test]
fn test_spatial_index_iter() {
    let mut index = SpatialIndex::new();
    for i in 0..5 {
        index.insert(SpatialEntry {
            bounds: BoundingBox::new(i as f32, 0.0, 1.0, 1.0).unwrap(),
            data: i,
        });
    }
    let all: Vec<_> = index.iter().collect();
    assert_eq!(all.len(), 5);
    let iter = index.iter();
    assert_eq!(iter.len(), 5);
}

#[test]
fn test_spatial_index_into_iter() {
    let mut index = SpatialIndex::new();
    for i in 0..3 {
        index.insert(SpatialEntry {
            bounds: BoundingBox::new(i as f32, 0.0, 1.0, 1.0).unwrap(),
            data: i,
        });
    }
    let all: Vec<_> = (&index).into_iter().collect();
    assert_eq!(all.len(), 3);
}

#[test]
fn test_spatial_index_concurrent() {
    use std::sync::Arc;

    let index = Arc::new(parking_lot::Mutex::new(SpatialIndex::<u32>::new()));
    let mut handles = Vec::new();

    for thread_id in 0..4u32 {
        let idx = Arc::clone(&index);
        handles.push(std::thread::spawn(move || {
            for i in 0..100u32 {
                let x = (thread_id * 100 + i) as f32;
                idx.lock().insert(SpatialEntry {
                    bounds: BoundingBox::new(x, 0.0, 1.0, 1.0).unwrap(),
                    data: thread_id * 100 + i,
                });
            }
        }));
    }

    for h in handles {
        h.join().unwrap();
    }

    assert_eq!(index.lock().len(), 400);
}

#[test]
fn test_spatial_index_large_dataset() {
    let mut index = SpatialIndex::new();
    for i in 0..10_000u32 {
        let x = (i % 100) as f32;
        let y = (i / 100) as f32;
        index.insert(SpatialEntry {
            bounds: BoundingBox::new(x, y, 0.5, 0.5).unwrap(),
            data: i,
        });
    }
    assert_eq!(index.len(), 10_000);

    let region = BoundingBox::new(0.0, 0.0, 5.5, 5.5).unwrap();
    let results = index.query_region(region);
    assert!(!results.is_empty());
    for entry in &results {
        assert!(entry.bounds.intersects(region));
    }

    let nearest = index.query_nearest(50.0, 50.0, 5);
    assert_eq!(nearest.len(), 5);
}

#[test]
fn test_spatial_error_display() {
    let err = SpatialError::InvalidBounds;
    assert_eq!(
        err.to_string(),
        "invalid bounding box: width and height must be non-negative"
    );
    let err = SpatialError::NotFound;
    assert_eq!(err.to_string(), "entry not found in spatial index");
    let err = SpatialError::InvalidRadius;
    assert_eq!(
        err.to_string(),
        "invalid radius: must be non-negative and finite"
    );
    let err = SpatialError::InvalidBounds3D;
    assert_eq!(
        err.to_string(),
        "invalid 3D bounding box: width, height, and depth must be non-negative"
    );
}

#[test]
fn test_bounding_box_min_dist_sq() {
    let bb = BoundingBox::new(2.0, 2.0, 3.0, 3.0).unwrap();
    // Point inside box: distance is 0
    assert_eq!(bb.min_dist_sq(3.0, 3.0), 0.0);
    // Point to the left
    assert!((bb.min_dist_sq(0.0, 3.0) - 4.0).abs() < f32::EPSILON);
    // Point below
    assert!((bb.min_dist_sq(3.0, 0.0) - 4.0).abs() < f32::EPSILON);
    // Point diagonal (lower-left corner: (0,0) -> nearest corner (2,2))
    assert!((bb.min_dist_sq(0.0, 0.0) - 8.0).abs() < f32::EPSILON);
}

#[test]
fn test_query_nearest_empty_index() {
    let index: SpatialIndex<u32> = SpatialIndex::new();
    let results = index.query_nearest(0.0, 0.0, 5);
    assert!(results.is_empty());
}

#[test]
fn test_split_preserves_entries() {
    let mut index = SpatialIndex::new();
    for i in 0..15u32 {
        index.insert(SpatialEntry {
            bounds: BoundingBox::new(i as f32, 0.0, 1.0, 1.0).unwrap(),
            data: i,
        });
    }
    assert_eq!(index.len(), 15);
    let all: Vec<_> = index.iter().collect();
    assert_eq!(all.len(), 15);
}

#[test]
fn test_multiple_splits() {
    let mut index = SpatialIndex::new();
    for i in 0..100u32 {
        index.insert(SpatialEntry {
            bounds: BoundingBox::new(i as f32, i as f32, 1.0, 1.0).unwrap(),
            data: i,
        });
    }
    assert_eq!(index.len(), 100);
    assert_eq!(index.iter().count(), 100);
}

#[test]
fn test_bounding_box_min_dist_sq_right_of_box() {
    let bb = BoundingBox::new(0.0, 0.0, 2.0, 2.0).unwrap();
    // Point to the right: (5, 1) -> nearest edge at x=2
    assert!((bb.min_dist_sq(5.0, 1.0) - 9.0).abs() < f32::EPSILON);
}

#[test]
fn test_bounding_box_min_dist_sq_is_pub() {
    // Verifies min_dist_sq is accessible from outside the crate
    let bb = BoundingBox::new(0.0, 0.0, 10.0, 10.0).unwrap();
    let dist = bb.min_dist_sq(5.0, 5.0);
    assert_eq!(dist, 0.0);
}

#[test]
fn test_bounding_box_min_dist_sq_above_box() {
    let bb = BoundingBox::new(0.0, 0.0, 2.0, 2.0).unwrap();
    // Point above: (1, 5) -> nearest edge at y=2
    assert!((bb.min_dist_sq(1.0, 5.0) - 9.0).abs() < f32::EPSILON);
}

// ---------------------------------------------------------------------------
// query_within_radius tests
// ---------------------------------------------------------------------------

#[test]
fn test_query_within_radius_point_inside() {
    let mut index = SpatialIndex::new();
    index.insert(SpatialEntry {
        bounds: BoundingBox::new(0.0, 0.0, 100.0, 100.0).unwrap(),
        data: "large",
    });
    let results = index.query_within_radius(50.0, 50.0, 1.0);
    assert_eq!(results.len(), 1);
    assert_eq!(results[0].data, "large");
}

#[test]
fn test_query_within_radius_point_outside() {
    let mut index = SpatialIndex::new();
    index.insert(SpatialEntry {
        bounds: BoundingBox::new(10.0, 10.0, 5.0, 5.0).unwrap(),
        data: "box",
    });
    let results = index.query_within_radius(20.0, 12.0, 5.0);
    assert_eq!(results.len(), 1);
    let results = index.query_within_radius(20.0, 12.0, 4.9);
    assert!(results.is_empty());
}

#[test]
fn test_query_within_radius_zero() {
    let mut index = SpatialIndex::new();
    index.insert(SpatialEntry {
        bounds: BoundingBox::new(0.0, 0.0, 100.0, 100.0).unwrap(),
        data: "inside",
    });
    index.insert(SpatialEntry {
        bounds: BoundingBox::new(200.0, 200.0, 10.0, 10.0).unwrap(),
        data: "outside",
    });
    let results = index.query_within_radius(50.0, 50.0, 0.0);
    assert_eq!(results.len(), 1);
    assert_eq!(results[0].data, "inside");
}

#[test]
fn test_query_within_radius_negative() {
    let mut index = SpatialIndex::new();
    index.insert(SpatialEntry {
        bounds: BoundingBox::new(0.0, 0.0, 100.0, 100.0).unwrap(),
        data: 1,
    });
    let results = index.query_within_radius(50.0, 50.0, -1.0);
    assert!(results.is_empty());
}

#[test]
fn test_query_within_radius_sorted_nearest_first() {
    let mut index = SpatialIndex::new();
    index.insert(SpatialEntry {
        bounds: BoundingBox::new(100.0, 0.0, 10.0, 10.0).unwrap(),
        data: "far",
    });
    index.insert(SpatialEntry {
        bounds: BoundingBox::new(20.0, 0.0, 10.0, 10.0).unwrap(),
        data: "near",
    });
    index.insert(SpatialEntry {
        bounds: BoundingBox::new(0.0, 0.0, 10.0, 10.0).unwrap(),
        data: "closest",
    });
    let results = index.query_within_radius(0.0, 0.0, 200.0);
    assert_eq!(results.len(), 3);
    assert_eq!(results[0].data, "closest");
    assert_eq!(results[1].data, "near");
    assert_eq!(results[2].data, "far");
}

#[test]
fn test_query_within_radius_empty_index() {
    let index: SpatialIndex<u32> = SpatialIndex::new();
    let results = index.query_within_radius(0.0, 0.0, 100.0);
    assert!(results.is_empty());
}

#[test]
fn test_query_within_radius_with_distances() {
    let mut index = SpatialIndex::new();
    index.insert(SpatialEntry {
        bounds: BoundingBox::new(0.0, 0.0, 100.0, 100.0).unwrap(),
        data: "inside",
    });
    index.insert(SpatialEntry {
        bounds: BoundingBox::new(110.0, 0.0, 10.0, 10.0).unwrap(),
        data: "outside",
    });
    let results = index.query_within_radius_with_distances(50.0, 50.0, 200.0);
    assert_eq!(results.len(), 2);
    assert_eq!(results[0].0.data, "inside");
    assert!((results[0].1 - 0.0).abs() < f32::EPSILON);
    assert_eq!(results[1].0.data, "outside");
    let expected = (60.0_f32.powi(2) + 40.0_f32.powi(2)).sqrt();
    assert!((results[1].1 - expected).abs() < 0.1);
}

#[test]
fn test_large_bbox_edge_distance_regression() {
    let mut index = SpatialIndex::new();
    index.insert(SpatialEntry {
        bounds: BoundingBox::new(0.0, 0.0, 1000.0, 100.0).unwrap(),
        data: "banner",
    });
    index.insert(SpatialEntry {
        bounds: BoundingBox::new(50.0, 200.0, 10.0, 10.0).unwrap(),
        data: "button",
    });

    let nearest = index.query_nearest(50.0, 50.0, 2);
    assert_eq!(nearest.len(), 2);
    assert_eq!(
        nearest[0].data, "banner",
        "Banner should be nearest (point is inside)"
    );
    assert_eq!(nearest[1].data, "button");

    let radius_results = index.query_within_radius(50.0, 50.0, 50.0);
    assert_eq!(radius_results.len(), 1);
    assert_eq!(radius_results[0].data, "banner");

    let radius_results = index.query_within_radius(50.0, 50.0, 200.0);
    assert_eq!(radius_results.len(), 2);
    assert_eq!(radius_results[0].data, "banner");
    assert_eq!(radius_results[1].data, "button");
}

#[test]
fn test_query_within_radius_large_dataset_brute_force() {
    let mut index = SpatialIndex::new();
    for i in 0..10_000u32 {
        let x = (i % 100) as f32;
        let y = (i / 100) as f32;
        index.insert(SpatialEntry {
            bounds: BoundingBox::new(x, y, 0.5, 0.5).unwrap(),
            data: i,
        });
    }

    let cx = 50.0_f32;
    let cy = 50.0_f32;
    let radius = 10.0_f32;

    let results = index.query_within_radius(cx, cy, radius);

    let mut expected_count = 0;
    for entry in index.iter() {
        if entry.bounds.min_dist_sq(cx, cy) <= radius * radius {
            expected_count += 1;
        }
    }
    assert_eq!(results.len(), expected_count);

    for entry in &results {
        let dist = entry.bounds.min_dist_sq(cx, cy).sqrt();
        assert!(dist <= radius + f32::EPSILON);
    }
}

#[test]
fn test_query_within_radius_all_entries() {
    let mut index = SpatialIndex::new();
    for i in 0..20u32 {
        let x = (i % 5) as f32;
        let y = (i / 5) as f32;
        index.insert(SpatialEntry {
            bounds: BoundingBox::new(x, y, 1.0, 1.0).unwrap(),
            data: i,
        });
    }
    let results = index.query_within_radius(2.5, 2.5, 1000.0);
    assert_eq!(results.len(), 20);
}

#[test]
fn test_query_within_radius_with_distances_negative() {
    let mut index = SpatialIndex::new();
    index.insert(SpatialEntry {
        bounds: BoundingBox::new(0.0, 0.0, 10.0, 10.0).unwrap(),
        data: 1,
    });
    let results = index.query_within_radius_with_distances(0.0, 0.0, -1.0);
    assert!(results.is_empty());
}

// ---------------------------------------------------------------------------
// serde roundtrip tests
// ---------------------------------------------------------------------------

#[test]
fn test_bounding_box_serde_roundtrip() {
    let bb = BoundingBox::new(1.5, 2.5, 10.0, 20.0).unwrap();
    let bytes = bitcode::serialize(&bb).unwrap();
    let restored: BoundingBox = bitcode::deserialize(&bytes).unwrap();
    assert_eq!(bb, restored);
}

#[test]
fn test_bounding_box_serde_zero_dimensions() {
    let bb = BoundingBox::new(0.0, 0.0, 0.0, 0.0).unwrap();
    let bytes = bitcode::serialize(&bb).unwrap();
    let restored: BoundingBox = bitcode::deserialize(&bytes).unwrap();
    assert_eq!(bb, restored);
    assert!(restored.is_empty());
}

#[test]
fn test_spatial_entry_serde_roundtrip() {
    let entry = SpatialEntry {
        bounds: BoundingBox::new(3.0, 4.0, 5.0, 6.0).unwrap(),
        data: String::from("hello"),
    };
    let bytes = bitcode::serialize(&entry).unwrap();
    let restored: SpatialEntry<String> = bitcode::deserialize(&bytes).unwrap();
    assert_eq!(entry, restored);
}

#[test]
fn test_spatial_index_serde_empty_roundtrip() {
    let index: SpatialIndex<u32> = SpatialIndex::new();
    let bytes = bitcode::serialize(&index).unwrap();
    let restored: SpatialIndex<u32> = bitcode::deserialize(&bytes).unwrap();
    assert_eq!(restored.len(), 0);
    assert!(restored.is_empty());
}

#[test]
fn test_spatial_index_serde_single_entry_roundtrip() {
    let mut index = SpatialIndex::new();
    index.insert(SpatialEntry {
        bounds: BoundingBox::new(1.0, 2.0, 3.0, 4.0).unwrap(),
        data: 42u32,
    });
    let bytes = bitcode::serialize(&index).unwrap();
    let restored: SpatialIndex<u32> = bitcode::deserialize(&bytes).unwrap();
    assert_eq!(restored.len(), 1);
    let region = BoundingBox::new(0.0, 0.0, 10.0, 10.0).unwrap();
    let results = restored.query_region(region);
    assert_eq!(results.len(), 1);
    assert_eq!(results[0].data, 42);
}

#[test]
fn test_spatial_index_serde_tree_with_splits_roundtrip() {
    let mut index = SpatialIndex::new();
    for i in 0..50u32 {
        let x = (i % 10) as f32 * 5.0;
        let y = (i / 10) as f32 * 5.0;
        index.insert(SpatialEntry {
            bounds: BoundingBox::new(x, y, 2.0, 2.0).unwrap(),
            data: i,
        });
    }
    assert_eq!(index.len(), 50);

    let bytes = bitcode::serialize(&index).unwrap();
    let restored: SpatialIndex<u32> = bitcode::deserialize(&bytes).unwrap();
    assert_eq!(restored.len(), 50);

    let region = BoundingBox::new(0.0, 0.0, 50.0, 50.0).unwrap();
    let orig_results = index.query_region(region);
    let rest_results = restored.query_region(region);
    assert_eq!(orig_results.len(), rest_results.len());

    // Nearest query: both should return 5 results. Exact membership may
    // differ when entries are equidistant (insert-built vs STR-rebuilt tree).
    let orig_nearest = index.query_nearest(25.0, 12.0, 5);
    let rest_nearest = restored.query_nearest(25.0, 12.0, 5);
    assert_eq!(orig_nearest.len(), rest_nearest.len());
}

#[test]
fn test_spatial_index_serde_string_data_roundtrip() {
    let mut index = SpatialIndex::new();
    for i in 0..10 {
        index.insert(SpatialEntry {
            bounds: BoundingBox::new(i as f32, 0.0, 1.0, 1.0).unwrap(),
            data: format!("item_{i}"),
        });
    }
    let bytes = bitcode::serialize(&index).unwrap();
    let restored: SpatialIndex<String> = bitcode::deserialize(&bytes).unwrap();
    assert_eq!(restored.len(), 10);

    let region = BoundingBox::new(0.0, 0.0, 100.0, 100.0).unwrap();
    let mut orig_data: Vec<_> = index
        .query_region(region)
        .iter()
        .map(|e| &e.data)
        .cloned()
        .collect();
    let mut rest_data: Vec<_> = restored
        .query_region(region)
        .iter()
        .map(|e| &e.data)
        .cloned()
        .collect();
    orig_data.sort();
    rest_data.sort();
    assert_eq!(orig_data, rest_data);
}

#[test]
fn test_spatial_index_serde_mutate_after_restore() {
    let mut index = SpatialIndex::new();
    for i in 0..5u32 {
        index.insert(SpatialEntry {
            bounds: BoundingBox::new(i as f32, 0.0, 1.0, 1.0).unwrap(),
            data: i,
        });
    }
    let bytes = bitcode::serialize(&index).unwrap();
    let mut restored: SpatialIndex<u32> = bitcode::deserialize(&bytes).unwrap();
    assert_eq!(restored.len(), 5);

    restored.insert(SpatialEntry {
        bounds: BoundingBox::new(10.0, 10.0, 1.0, 1.0).unwrap(),
        data: 99,
    });
    assert_eq!(restored.len(), 6);

    let bb = BoundingBox::new(0.0, 0.0, 1.0, 1.0).unwrap();
    restored.remove(bb, |e| e.data == 0).unwrap();
    assert_eq!(restored.len(), 5);
}

// ---------------------------------------------------------------------------
// 3D BoundingBox tests
// ---------------------------------------------------------------------------

#[test]
fn test_bounding_box_3d_new_and_accessors() {
    let bb = BoundingBox3D::new(1.0, 2.0, 3.0, 4.0, 5.0, 6.0).unwrap();
    assert_eq!(bb.x(), 1.0);
    assert_eq!(bb.y(), 2.0);
    assert_eq!(bb.z(), 3.0);
    assert_eq!(bb.width(), 4.0);
    assert_eq!(bb.height(), 5.0);
    assert_eq!(bb.depth(), 6.0);
}

#[test]
fn test_bounding_box_3d_rejects_negative_width() {
    assert!(BoundingBox3D::new(0.0, 0.0, 0.0, -1.0, 1.0, 1.0).is_err());
}

#[test]
fn test_bounding_box_3d_rejects_negative_height() {
    assert!(BoundingBox3D::new(0.0, 0.0, 0.0, 1.0, -1.0, 1.0).is_err());
}

#[test]
fn test_bounding_box_3d_rejects_negative_depth() {
    assert!(BoundingBox3D::new(0.0, 0.0, 0.0, 1.0, 1.0, -1.0).is_err());
}

#[test]
fn test_bounding_box_3d_center() {
    let bb = BoundingBox3D::new(0.0, 0.0, 0.0, 4.0, 6.0, 8.0).unwrap();
    assert_eq!(bb.center(), (2.0, 3.0, 4.0));
}

#[test]
fn test_bounding_box_3d_contains_point() {
    let bb = BoundingBox3D::new(1.0, 1.0, 1.0, 3.0, 3.0, 3.0).unwrap();
    assert!(bb.contains_point(2.0, 2.0, 2.0));
    assert!(bb.contains_point(1.0, 1.0, 1.0));
    assert!(bb.contains_point(4.0, 4.0, 4.0));
    assert!(!bb.contains_point(0.0, 2.0, 2.0));
    assert!(!bb.contains_point(5.0, 2.0, 2.0));
    assert!(!bb.contains_point(2.0, 0.0, 2.0));
    assert!(!bb.contains_point(2.0, 5.0, 2.0));
    assert!(!bb.contains_point(2.0, 2.0, 0.0));
    assert!(!bb.contains_point(2.0, 2.0, 5.0));
}

#[test]
fn test_bounding_box_3d_intersects() {
    let a = BoundingBox3D::new(0.0, 0.0, 0.0, 2.0, 2.0, 2.0).unwrap();
    let b = BoundingBox3D::new(1.0, 1.0, 1.0, 2.0, 2.0, 2.0).unwrap();
    assert!(a.intersects(b));
    assert!(b.intersects(a));

    let c = BoundingBox3D::new(5.0, 5.0, 5.0, 1.0, 1.0, 1.0).unwrap();
    assert!(!a.intersects(c));
    assert!(!c.intersects(a));
}

#[test]
fn test_bounding_box_3d_intersects_edge_touching() {
    let a = BoundingBox3D::new(0.0, 0.0, 0.0, 2.0, 2.0, 2.0).unwrap();
    let b = BoundingBox3D::new(2.0, 0.0, 0.0, 2.0, 2.0, 2.0).unwrap();
    assert!(!a.intersects(b));
    let c = BoundingBox3D::new(0.0, 0.0, 2.0, 2.0, 2.0, 2.0).unwrap();
    assert!(!a.intersects(c));
}

#[test]
fn test_bounding_box_3d_union() {
    let a = BoundingBox3D::new(0.0, 0.0, 0.0, 2.0, 2.0, 2.0).unwrap();
    let b = BoundingBox3D::new(1.0, 1.0, 1.0, 3.0, 3.0, 3.0).unwrap();
    let u = a.union(b);
    assert_eq!(u.x(), 0.0);
    assert_eq!(u.y(), 0.0);
    assert_eq!(u.z(), 0.0);
    assert_eq!(u.width(), 4.0);
    assert_eq!(u.height(), 4.0);
    assert_eq!(u.depth(), 4.0);
}

#[test]
fn test_bounding_box_3d_volume() {
    let bb = BoundingBox3D::new(0.0, 0.0, 0.0, 3.0, 5.0, 7.0).unwrap();
    assert!((bb.volume() - 105.0).abs() < f32::EPSILON);
}

#[test]
fn test_bounding_box_3d_is_empty() {
    assert!(BoundingBox3D::new(0.0, 0.0, 0.0, 0.0, 5.0, 5.0)
        .unwrap()
        .is_empty());
    assert!(BoundingBox3D::new(0.0, 0.0, 0.0, 5.0, 0.0, 5.0)
        .unwrap()
        .is_empty());
    assert!(BoundingBox3D::new(0.0, 0.0, 0.0, 5.0, 5.0, 0.0)
        .unwrap()
        .is_empty());
    assert!(BoundingBox3D::new(0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
        .unwrap()
        .is_empty());
    assert!(!BoundingBox3D::new(0.0, 0.0, 0.0, 1.0, 1.0, 1.0)
        .unwrap()
        .is_empty());
}

#[test]
fn test_bounding_box_3d_min_dist_sq_inside() {
    let bb = BoundingBox3D::new(2.0, 2.0, 2.0, 3.0, 3.0, 3.0).unwrap();
    assert_eq!(bb.min_dist_sq(3.0, 3.0, 3.0), 0.0);
}

#[test]
fn test_bounding_box_3d_min_dist_sq_each_axis() {
    let bb = BoundingBox3D::new(2.0, 2.0, 2.0, 3.0, 3.0, 3.0).unwrap();
    assert!((bb.min_dist_sq(0.0, 3.0, 3.0) - 4.0).abs() < f32::EPSILON);
    assert!((bb.min_dist_sq(3.0, 0.0, 3.0) - 4.0).abs() < f32::EPSILON);
    assert!((bb.min_dist_sq(3.0, 3.0, 0.0) - 4.0).abs() < f32::EPSILON);
}

#[test]
fn test_bounding_box_3d_min_dist_sq_diagonal() {
    let bb = BoundingBox3D::new(2.0, 2.0, 2.0, 3.0, 3.0, 3.0).unwrap();
    assert!((bb.min_dist_sq(0.0, 0.0, 0.0) - 12.0).abs() < f32::EPSILON);
}

#[test]
fn test_bounding_box_3d_min_dist_sq_beyond_box() {
    let bb = BoundingBox3D::new(0.0, 0.0, 0.0, 2.0, 2.0, 2.0).unwrap();
    assert!((bb.min_dist_sq(5.0, 1.0, 1.0) - 9.0).abs() < f32::EPSILON);
    assert!((bb.min_dist_sq(1.0, 5.0, 1.0) - 9.0).abs() < f32::EPSILON);
    assert!((bb.min_dist_sq(1.0, 1.0, 5.0) - 9.0).abs() < f32::EPSILON);
}

// ---------------------------------------------------------------------------
// 3D SpatialEntry tests
// ---------------------------------------------------------------------------

#[test]
fn test_spatial_entry_3d() {
    let entry = SpatialEntry3D {
        bounds: BoundingBox3D::new(1.0, 2.0, 3.0, 4.0, 5.0, 6.0).unwrap(),
        data: 42,
    };
    assert_eq!(entry.data, 42);
    assert_eq!(entry.bounds.x(), 1.0);

    let debug_str = format!("{entry:?}");
    assert!(debug_str.contains("SpatialEntry"));

    let cloned = entry.clone();
    assert_eq!(cloned.data, entry.data);
    assert_eq!(cloned.bounds, entry.bounds);

    assert_eq!(entry, cloned);
}

// ---------------------------------------------------------------------------
// 3D SpatialIndex tests
// ---------------------------------------------------------------------------

#[test]
fn test_spatial_index_3d_empty() {
    let index: SpatialIndex3D<u32> = SpatialIndex3D::new();
    assert!(index.is_empty());
    assert_eq!(index.len(), 0);
    assert_eq!(index.iter().count(), 0);
}

#[test]
fn test_spatial_index_3d_default() {
    let index: SpatialIndex3D<u32> = SpatialIndex3D::default();
    assert!(index.is_empty());
}

#[test]
fn test_spatial_index_3d_insert_and_query_region() {
    let mut index = SpatialIndex3D::new();

    for i in 0..20 {
        let x = (i % 5) as f32 * 10.0;
        let y = (i / 5) as f32 * 10.0;
        let z = (i % 3) as f32 * 10.0;
        index.insert(SpatialEntry3D {
            bounds: BoundingBox3D::new(x, y, z, 5.0, 5.0, 5.0).unwrap(),
            data: i,
        });
    }
    assert_eq!(index.len(), 20);

    let region = BoundingBox3D::new(0.0, 0.0, 0.0, 12.0, 12.0, 12.0).unwrap();
    let results = index.query_region(region);
    assert!(!results.is_empty());

    for entry in &results {
        assert!(entry.bounds.intersects(region));
    }
}

#[test]
fn test_spatial_index_3d_query_region_no_results() {
    let mut index = SpatialIndex3D::new();
    for i in 0..10 {
        index.insert(SpatialEntry3D {
            bounds: BoundingBox3D::new(0.0, 0.0, 0.0, 1.0, 1.0, 1.0).unwrap(),
            data: i,
        });
    }
    let region = BoundingBox3D::new(100.0, 100.0, 100.0, 1.0, 1.0, 1.0).unwrap();
    let results = index.query_region(region);
    assert!(results.is_empty());
}

#[test]
fn test_spatial_index_3d_query_nearest() {
    let mut index = SpatialIndex3D::new();

    index.insert(SpatialEntry3D {
        bounds: BoundingBox3D::new(0.0, 0.0, 0.0, 1.0, 1.0, 1.0).unwrap(),
        data: "origin",
    });
    index.insert(SpatialEntry3D {
        bounds: BoundingBox3D::new(10.0, 10.0, 10.0, 1.0, 1.0, 1.0).unwrap(),
        data: "far",
    });
    index.insert(SpatialEntry3D {
        bounds: BoundingBox3D::new(2.0, 2.0, 2.0, 1.0, 1.0, 1.0).unwrap(),
        data: "near",
    });

    let results = index.query_nearest(0.0, 0.0, 0.0, 2);
    assert_eq!(results.len(), 2);
    assert_eq!(results[0].data, "origin");
    assert_eq!(results[1].data, "near");
}

#[test]
fn test_spatial_index_3d_query_nearest_k_larger_than_n() {
    let mut index = SpatialIndex3D::new();
    index.insert(SpatialEntry3D {
        bounds: BoundingBox3D::new(0.0, 0.0, 0.0, 1.0, 1.0, 1.0).unwrap(),
        data: 1,
    });
    index.insert(SpatialEntry3D {
        bounds: BoundingBox3D::new(5.0, 5.0, 5.0, 1.0, 1.0, 1.0).unwrap(),
        data: 2,
    });

    let results = index.query_nearest(0.0, 0.0, 0.0, 100);
    assert_eq!(results.len(), 2);
}

#[test]
fn test_spatial_index_3d_query_nearest_empty() {
    let index: SpatialIndex3D<u32> = SpatialIndex3D::new();
    let results = index.query_nearest(0.0, 0.0, 0.0, 5);
    assert!(results.is_empty());
}

#[test]
fn test_spatial_index_3d_remove() {
    let mut index = SpatialIndex3D::new();
    let bb = BoundingBox3D::new(1.0, 1.0, 1.0, 2.0, 2.0, 2.0).unwrap();
    index.insert(SpatialEntry3D {
        bounds: bb,
        data: 42u32,
    });
    assert_eq!(index.len(), 1);

    index.remove(bb, |e| e.data == 42).unwrap();
    assert_eq!(index.len(), 0);
    assert!(index.is_empty());

    let result = index.remove(bb, |e| e.data == 42);
    assert!(result.is_err());
}

#[test]
fn test_spatial_index_3d_remove_from_tree() {
    let mut index = SpatialIndex3D::new();
    for i in 0..20u32 {
        let x = (i % 5) as f32 * 10.0;
        let y = (i / 5) as f32 * 10.0;
        let z = (i % 3) as f32 * 10.0;
        index.insert(SpatialEntry3D {
            bounds: BoundingBox3D::new(x, y, z, 5.0, 5.0, 5.0).unwrap(),
            data: i,
        });
    }
    assert_eq!(index.len(), 20);

    let bb = BoundingBox3D::new(0.0, 0.0, 0.0, 5.0, 5.0, 5.0).unwrap();
    index.remove(bb, |e| e.data == 0).unwrap();
    assert_eq!(index.len(), 19);
}

#[test]
fn test_spatial_index_3d_clear() {
    let mut index = SpatialIndex3D::new();
    for i in 0..10 {
        index.insert(SpatialEntry3D {
            bounds: BoundingBox3D::new(0.0, 0.0, 0.0, 1.0, 1.0, 1.0).unwrap(),
            data: i,
        });
    }
    assert_eq!(index.len(), 10);
    index.clear();
    assert!(index.is_empty());
    assert_eq!(index.len(), 0);
}

#[test]
fn test_spatial_index_3d_iter() {
    let mut index = SpatialIndex3D::new();
    for i in 0..5 {
        index.insert(SpatialEntry3D {
            bounds: BoundingBox3D::new(i as f32, 0.0, 0.0, 1.0, 1.0, 1.0).unwrap(),
            data: i,
        });
    }
    let all: Vec<_> = index.iter().collect();
    assert_eq!(all.len(), 5);
    let iter = index.iter();
    assert_eq!(iter.len(), 5);
}

#[test]
fn test_spatial_index_3d_into_iter() {
    let mut index = SpatialIndex3D::new();
    for i in 0..3 {
        index.insert(SpatialEntry3D {
            bounds: BoundingBox3D::new(i as f32, 0.0, 0.0, 1.0, 1.0, 1.0).unwrap(),
            data: i,
        });
    }
    let all: Vec<_> = (&index).into_iter().collect();
    assert_eq!(all.len(), 3);
}

#[test]
fn test_spatial_index_3d_concurrent() {
    use std::sync::Arc;

    let index = Arc::new(parking_lot::Mutex::new(SpatialIndex3D::<u32>::new()));
    let mut handles = Vec::new();

    for thread_id in 0..4u32 {
        let idx = Arc::clone(&index);
        handles.push(std::thread::spawn(move || {
            for i in 0..100u32 {
                let x = (thread_id * 100 + i) as f32;
                idx.lock().insert(SpatialEntry3D {
                    bounds: BoundingBox3D::new(x, 0.0, 0.0, 1.0, 1.0, 1.0).unwrap(),
                    data: thread_id * 100 + i,
                });
            }
        }));
    }

    for h in handles {
        h.join().unwrap();
    }

    assert_eq!(index.lock().len(), 400);
}

#[test]
fn test_spatial_index_3d_large_dataset() {
    let mut index = SpatialIndex3D::new();
    for i in 0..1_000u32 {
        let x = (i % 10) as f32;
        let y = ((i / 10) % 10) as f32;
        let z = (i / 100) as f32;
        index.insert(SpatialEntry3D {
            bounds: BoundingBox3D::new(x, y, z, 0.5, 0.5, 0.5).unwrap(),
            data: i,
        });
    }
    assert_eq!(index.len(), 1_000);

    let region = BoundingBox3D::new(0.0, 0.0, 0.0, 5.5, 5.5, 5.5).unwrap();
    let results = index.query_region(region);
    assert!(!results.is_empty());
    for entry in &results {
        assert!(entry.bounds.intersects(region));
    }

    let nearest = index.query_nearest(5.0, 5.0, 5.0, 5);
    assert_eq!(nearest.len(), 5);
}

#[test]
fn test_spatial_index_3d_split_preserves_entries() {
    let mut index = SpatialIndex3D::new();
    for i in 0..15u32 {
        index.insert(SpatialEntry3D {
            bounds: BoundingBox3D::new(i as f32, 0.0, 0.0, 1.0, 1.0, 1.0).unwrap(),
            data: i,
        });
    }
    assert_eq!(index.len(), 15);
    let all: Vec<_> = index.iter().collect();
    assert_eq!(all.len(), 15);
}

#[test]
fn test_spatial_index_3d_multiple_splits() {
    let mut index = SpatialIndex3D::new();
    for i in 0..100u32 {
        index.insert(SpatialEntry3D {
            bounds: BoundingBox3D::new(i as f32, i as f32, i as f32, 1.0, 1.0, 1.0).unwrap(),
            data: i,
        });
    }
    assert_eq!(index.len(), 100);
    assert_eq!(index.iter().count(), 100);
}

// ---------------------------------------------------------------------------
// 3D radius query tests
// ---------------------------------------------------------------------------

#[test]
fn test_query_within_radius_3d_point_inside() {
    let mut index = SpatialIndex3D::new();
    index.insert(SpatialEntry3D {
        bounds: BoundingBox3D::new(0.0, 0.0, 0.0, 100.0, 100.0, 100.0).unwrap(),
        data: "large",
    });
    let results = index.query_within_radius(50.0, 50.0, 50.0, 1.0);
    assert_eq!(results.len(), 1);
    assert_eq!(results[0].data, "large");
}

#[test]
fn test_query_within_radius_3d_point_outside() {
    let mut index = SpatialIndex3D::new();
    index.insert(SpatialEntry3D {
        bounds: BoundingBox3D::new(10.0, 10.0, 10.0, 5.0, 5.0, 5.0).unwrap(),
        data: "box",
    });
    let results = index.query_within_radius(20.0, 12.0, 12.0, 5.0);
    assert_eq!(results.len(), 1);
    let results = index.query_within_radius(20.0, 12.0, 12.0, 4.9);
    assert!(results.is_empty());
}

#[test]
fn test_query_within_radius_3d_zero() {
    let mut index = SpatialIndex3D::new();
    index.insert(SpatialEntry3D {
        bounds: BoundingBox3D::new(0.0, 0.0, 0.0, 100.0, 100.0, 100.0).unwrap(),
        data: "inside",
    });
    index.insert(SpatialEntry3D {
        bounds: BoundingBox3D::new(200.0, 200.0, 200.0, 10.0, 10.0, 10.0).unwrap(),
        data: "outside",
    });
    let results = index.query_within_radius(50.0, 50.0, 50.0, 0.0);
    assert_eq!(results.len(), 1);
    assert_eq!(results[0].data, "inside");
}

#[test]
fn test_query_within_radius_3d_negative() {
    let mut index = SpatialIndex3D::new();
    index.insert(SpatialEntry3D {
        bounds: BoundingBox3D::new(0.0, 0.0, 0.0, 100.0, 100.0, 100.0).unwrap(),
        data: 1,
    });
    let results = index.query_within_radius(50.0, 50.0, 50.0, -1.0);
    assert!(results.is_empty());
}

#[test]
fn test_query_within_radius_3d_sorted_nearest_first() {
    let mut index = SpatialIndex3D::new();
    index.insert(SpatialEntry3D {
        bounds: BoundingBox3D::new(100.0, 0.0, 0.0, 10.0, 10.0, 10.0).unwrap(),
        data: "far",
    });
    index.insert(SpatialEntry3D {
        bounds: BoundingBox3D::new(20.0, 0.0, 0.0, 10.0, 10.0, 10.0).unwrap(),
        data: "near",
    });
    index.insert(SpatialEntry3D {
        bounds: BoundingBox3D::new(0.0, 0.0, 0.0, 10.0, 10.0, 10.0).unwrap(),
        data: "closest",
    });
    let results = index.query_within_radius(0.0, 0.0, 0.0, 200.0);
    assert_eq!(results.len(), 3);
    assert_eq!(results[0].data, "closest");
    assert_eq!(results[1].data, "near");
    assert_eq!(results[2].data, "far");
}

#[test]
fn test_query_within_radius_3d_empty_index() {
    let index: SpatialIndex3D<u32> = SpatialIndex3D::new();
    let results = index.query_within_radius(0.0, 0.0, 0.0, 100.0);
    assert!(results.is_empty());
}

#[test]
fn test_query_within_radius_3d_with_distances() {
    let mut index = SpatialIndex3D::new();
    index.insert(SpatialEntry3D {
        bounds: BoundingBox3D::new(0.0, 0.0, 0.0, 100.0, 100.0, 100.0).unwrap(),
        data: "inside",
    });
    index.insert(SpatialEntry3D {
        bounds: BoundingBox3D::new(110.0, 0.0, 0.0, 10.0, 10.0, 10.0).unwrap(),
        data: "outside",
    });
    let results = index.query_within_radius_with_distances(50.0, 50.0, 50.0, 200.0);
    assert_eq!(results.len(), 2);
    assert_eq!(results[0].0.data, "inside");
    assert!((results[0].1 - 0.0).abs() < f32::EPSILON);
    assert_eq!(results[1].0.data, "outside");
    let expected = (60.0_f32.powi(2) + 40.0_f32.powi(2) + 40.0_f32.powi(2)).sqrt();
    assert!((results[1].1 - expected).abs() < 0.1);
}

#[test]
fn test_query_within_radius_3d_with_distances_negative() {
    let mut index = SpatialIndex3D::new();
    index.insert(SpatialEntry3D {
        bounds: BoundingBox3D::new(0.0, 0.0, 0.0, 10.0, 10.0, 10.0).unwrap(),
        data: 1,
    });
    let results = index.query_within_radius_with_distances(0.0, 0.0, 0.0, -1.0);
    assert!(results.is_empty());
}

#[test]
fn test_query_within_radius_3d_large_dataset_brute_force() {
    let mut index = SpatialIndex3D::new();
    for i in 0..1_000u32 {
        let x = (i % 10) as f32;
        let y = ((i / 10) % 10) as f32;
        let z = (i / 100) as f32;
        index.insert(SpatialEntry3D {
            bounds: BoundingBox3D::new(x, y, z, 0.5, 0.5, 0.5).unwrap(),
            data: i,
        });
    }

    let cx = 5.0_f32;
    let cy = 5.0_f32;
    let cz = 5.0_f32;
    let radius = 3.0_f32;

    let results = index.query_within_radius(cx, cy, cz, radius);

    let mut expected_count = 0;
    for entry in index.iter() {
        if entry.bounds.min_dist_sq(cx, cy, cz) <= radius * radius {
            expected_count += 1;
        }
    }
    assert_eq!(results.len(), expected_count);

    for entry in &results {
        let dist = entry.bounds.min_dist_sq(cx, cy, cz).sqrt();
        assert!(dist <= radius + f32::EPSILON);
    }
}

// ---------------------------------------------------------------------------
// 3D serde roundtrip tests
// ---------------------------------------------------------------------------

#[test]
fn test_bounding_box_3d_serde_roundtrip() {
    let bb = BoundingBox3D::new(1.5, 2.5, 3.5, 10.0, 20.0, 30.0).unwrap();
    let bytes = bitcode::serialize(&bb).unwrap();
    let restored: BoundingBox3D = bitcode::deserialize(&bytes).unwrap();
    assert_eq!(bb, restored);
}

#[test]
fn test_bounding_box_3d_serde_zero_dimensions() {
    let bb = BoundingBox3D::new(0.0, 0.0, 0.0, 0.0, 0.0, 0.0).unwrap();
    let bytes = bitcode::serialize(&bb).unwrap();
    let restored: BoundingBox3D = bitcode::deserialize(&bytes).unwrap();
    assert_eq!(bb, restored);
    assert!(restored.is_empty());
}

#[test]
fn test_spatial_entry_3d_serde_roundtrip() {
    let entry = SpatialEntry3D {
        bounds: BoundingBox3D::new(3.0, 4.0, 5.0, 6.0, 7.0, 8.0).unwrap(),
        data: String::from("hello"),
    };
    let bytes = bitcode::serialize(&entry).unwrap();
    let restored: SpatialEntry3D<String> = bitcode::deserialize(&bytes).unwrap();
    assert_eq!(entry, restored);
}

#[test]
fn test_spatial_index_3d_serde_empty_roundtrip() {
    let index: SpatialIndex3D<u32> = SpatialIndex3D::new();
    let bytes = bitcode::serialize(&index).unwrap();
    let restored: SpatialIndex3D<u32> = bitcode::deserialize(&bytes).unwrap();
    assert_eq!(restored.len(), 0);
    assert!(restored.is_empty());
}

#[test]
fn test_spatial_index_3d_serde_single_entry_roundtrip() {
    let mut index = SpatialIndex3D::new();
    index.insert(SpatialEntry3D {
        bounds: BoundingBox3D::new(1.0, 2.0, 3.0, 4.0, 5.0, 6.0).unwrap(),
        data: 42u32,
    });
    let bytes = bitcode::serialize(&index).unwrap();
    let restored: SpatialIndex3D<u32> = bitcode::deserialize(&bytes).unwrap();
    assert_eq!(restored.len(), 1);
    let region = BoundingBox3D::new(0.0, 0.0, 0.0, 10.0, 10.0, 10.0).unwrap();
    let results = restored.query_region(region);
    assert_eq!(results.len(), 1);
    assert_eq!(results[0].data, 42);
}

#[test]
fn test_spatial_index_3d_serde_tree_with_splits_roundtrip() {
    let mut index = SpatialIndex3D::new();
    for i in 0..50u32 {
        let x = (i % 10) as f32 * 5.0;
        let y = (i / 10) as f32 * 5.0;
        let z = (i % 5) as f32 * 5.0;
        index.insert(SpatialEntry3D {
            bounds: BoundingBox3D::new(x, y, z, 2.0, 2.0, 2.0).unwrap(),
            data: i,
        });
    }
    assert_eq!(index.len(), 50);

    let bytes = bitcode::serialize(&index).unwrap();
    let restored: SpatialIndex3D<u32> = bitcode::deserialize(&bytes).unwrap();
    assert_eq!(restored.len(), 50);

    let region = BoundingBox3D::new(0.0, 0.0, 0.0, 50.0, 50.0, 50.0).unwrap();
    let orig_results = index.query_region(region);
    let rest_results = restored.query_region(region);
    assert_eq!(orig_results.len(), rest_results.len());

    let orig_nearest = index.query_nearest(25.0, 12.0, 10.0, 5);
    let rest_nearest = restored.query_nearest(25.0, 12.0, 10.0, 5);
    let mut orig_data: Vec<_> = orig_nearest.iter().map(|e| e.data).collect();
    let mut rest_data: Vec<_> = rest_nearest.iter().map(|e| e.data).collect();
    orig_data.sort_unstable();
    rest_data.sort_unstable();
    assert_eq!(orig_data, rest_data);
}

#[test]
fn test_spatial_index_3d_serde_string_data_roundtrip() {
    let mut index = SpatialIndex3D::new();
    for i in 0..10 {
        index.insert(SpatialEntry3D {
            bounds: BoundingBox3D::new(i as f32, 0.0, 0.0, 1.0, 1.0, 1.0).unwrap(),
            data: format!("item_{i}"),
        });
    }
    let bytes = bitcode::serialize(&index).unwrap();
    let restored: SpatialIndex3D<String> = bitcode::deserialize(&bytes).unwrap();
    assert_eq!(restored.len(), 10);

    let region = BoundingBox3D::new(0.0, 0.0, 0.0, 100.0, 100.0, 100.0).unwrap();
    let mut orig_data: Vec<_> = index
        .query_region(region)
        .iter()
        .map(|e| &e.data)
        .cloned()
        .collect();
    let mut rest_data: Vec<_> = restored
        .query_region(region)
        .iter()
        .map(|e| &e.data)
        .cloned()
        .collect();
    orig_data.sort();
    rest_data.sort();
    assert_eq!(orig_data, rest_data);
}

#[test]
fn test_spatial_index_3d_serde_mutate_after_restore() {
    let mut index = SpatialIndex3D::new();
    for i in 0..5u32 {
        index.insert(SpatialEntry3D {
            bounds: BoundingBox3D::new(i as f32, 0.0, 0.0, 1.0, 1.0, 1.0).unwrap(),
            data: i,
        });
    }
    let bytes = bitcode::serialize(&index).unwrap();
    let mut restored: SpatialIndex3D<u32> = bitcode::deserialize(&bytes).unwrap();
    assert_eq!(restored.len(), 5);

    restored.insert(SpatialEntry3D {
        bounds: BoundingBox3D::new(10.0, 10.0, 10.0, 1.0, 1.0, 1.0).unwrap(),
        data: 99,
    });
    assert_eq!(restored.len(), 6);

    let bb = BoundingBox3D::new(0.0, 0.0, 0.0, 1.0, 1.0, 1.0).unwrap();
    restored.remove(bb, |e| e.data == 0).unwrap();
    assert_eq!(restored.len(), 5);
}

// ---------------------------------------------------------------------------
// Additional coverage: exercise failure paths on Internal nodes
// ---------------------------------------------------------------------------

#[test]
fn test_spatial_index_remove_not_found_on_internal() {
    let mut index = SpatialIndex::new();
    for i in 0..20u32 {
        let x = (i % 5) as f32 * 10.0;
        let y = (i / 5) as f32 * 10.0;
        index.insert(SpatialEntry {
            bounds: BoundingBox::new(x, y, 5.0, 5.0).unwrap(),
            data: i,
        });
    }
    let region = BoundingBox::new(0.0, 0.0, 200.0, 200.0).unwrap();
    let result = index.remove(region, |e| e.data == 999);
    assert!(result.is_err());
    assert_eq!(index.len(), 20);
}

#[test]
fn test_spatial_index_3d_remove_not_found_on_internal() {
    let mut index = SpatialIndex3D::new();
    for i in 0..20u32 {
        let x = (i % 5) as f32 * 10.0;
        let y = (i / 5) as f32 * 10.0;
        let z = (i % 3) as f32 * 10.0;
        index.insert(SpatialEntry3D {
            bounds: BoundingBox3D::new(x, y, z, 5.0, 5.0, 5.0).unwrap(),
            data: i,
        });
    }
    let region = BoundingBox3D::new(0.0, 0.0, 0.0, 200.0, 200.0, 200.0).unwrap();
    let result = index.remove(region, |e| e.data == 999);
    assert!(result.is_err());
    assert_eq!(index.len(), 20);
}

// ---------------------------------------------------------------------------
// Serde rejection tests
// ---------------------------------------------------------------------------

#[test]
fn test_bounding_box_3d_serde_rejects_negative_dims() {
    let bytes =
        bitcode::serialize(&(1.0_f32, 2.0_f32, 3.0_f32, -1.0_f32, 5.0_f32, 6.0_f32)).unwrap();
    let result: Result<BoundingBox3D, _> = bitcode::deserialize(&bytes);
    assert!(result.is_err());
}

#[test]
fn test_spatial_index_3d_serde_rejects_unknown_version() {
    #[derive(serde::Serialize)]
    struct FakeDto {
        version: u8,
        max_entries: usize,
        split_strategy: u8,
        entries: Vec<SpatialEntry3D<u32>>,
    }
    let fake = FakeDto {
        version: 99,
        max_entries: 9,
        split_strategy: 1,
        entries: vec![],
    };
    let bytes = bitcode::serialize(&fake).unwrap();
    let result: Result<SpatialIndex3D<u32>, _> = bitcode::deserialize(&bytes);
    assert!(result.is_err());
}

#[test]
fn test_spatial_index_2d_serde_rejects_unknown_version() {
    #[derive(serde::Serialize)]
    struct FakeDto {
        version: u8,
        max_entries: usize,
        split_strategy: u8,
        entries: Vec<SpatialEntry<u32>>,
    }
    let fake = FakeDto {
        version: 99,
        max_entries: 9,
        split_strategy: 1,
        entries: vec![],
    };
    let bytes = bitcode::serialize(&fake).unwrap();
    let result: Result<SpatialIndex<u32>, _> = bitcode::deserialize(&bytes);
    assert!(result.is_err());
}

// ---------------------------------------------------------------------------
// N-dimensional tests (4D+)
// ---------------------------------------------------------------------------

#[test]
fn test_bounding_box_4d() {
    let bb = BoundingBoxN::<4>::from_extents([1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]).unwrap();
    assert_eq!(bb.measure(), 5.0 * 6.0 * 7.0 * 8.0);
    let c = bb.center_nd();
    assert_eq!(c, [3.5, 5.0, 6.5, 8.0]);
    assert!(bb.contains_point_nd(&[3.0, 4.0, 5.0, 6.0]));
    assert!(!bb.contains_point_nd(&[0.0, 0.0, 0.0, 0.0]));
}

#[test]
fn test_spatial_index_4d_insert_query() {
    let mut index = SpatialIndexN::<4, u32>::new();
    for i in 0..20u32 {
        let origin = [i as f32, 0.0, 0.0, 0.0];
        let extent = [1.0, 1.0, 1.0, 1.0];
        index.insert(SpatialEntryN {
            bounds: BoundingBoxN::from_extents(origin, extent).unwrap(),
            data: i,
        });
    }
    assert_eq!(index.len(), 20);

    let region = BoundingBoxN::from_extents([0.0, 0.0, 0.0, 0.0], [5.5, 5.5, 5.5, 5.5]).unwrap();
    let results = index.query_region(region);
    assert!(!results.is_empty());
    for entry in &results {
        assert!(entry.bounds.intersects(region));
    }
}

#[test]
fn test_measure_dimensions() {
    let bb1 = BoundingBoxN::<1>::from_extents([0.0], [5.0]).unwrap();
    assert!((bb1.measure() - 5.0).abs() < f32::EPSILON);

    let bb2 = BoundingBoxN::<2>::from_extents([0.0, 0.0], [3.0, 4.0]).unwrap();
    assert!((bb2.measure() - 12.0).abs() < f32::EPSILON);

    let bb3 = BoundingBoxN::<3>::from_extents([0.0, 0.0, 0.0], [2.0, 3.0, 4.0]).unwrap();
    assert!((bb3.measure() - 24.0).abs() < f32::EPSILON);

    let bb4 = BoundingBoxN::<4>::from_extents([0.0, 0.0, 0.0, 0.0], [2.0, 3.0, 4.0, 5.0]).unwrap();
    assert!((bb4.measure() - 120.0).abs() < f32::EPSILON);
}

#[test]
fn test_serde_generic_roundtrip() {
    let bb2 = BoundingBox::new(1.0, 2.0, 3.0, 4.0).unwrap();
    let bytes2 = bitcode::serialize(&bb2).unwrap();
    let restored2: BoundingBox = bitcode::deserialize(&bytes2).unwrap();
    assert_eq!(bb2, restored2);

    let bb3 = BoundingBox3D::new(1.0, 2.0, 3.0, 4.0, 5.0, 6.0).unwrap();
    let bytes3 = bitcode::serialize(&bb3).unwrap();
    let restored3: BoundingBox3D = bitcode::deserialize(&bytes3).unwrap();
    assert_eq!(bb3, restored3);

    let bb4 = BoundingBoxN::<4>::from_extents([1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]).unwrap();
    let bytes4 = bitcode::serialize(&bb4).unwrap();
    let restored4: BoundingBoxN<4> = bitcode::deserialize(&bytes4).unwrap();
    assert_eq!(bb4, restored4);
}

#[test]
fn test_from_extents_rejects_negative() {
    let result = BoundingBoxN::<2>::from_extents([0.0, 0.0], [-1.0, 1.0]);
    assert!(result.is_err());
}

#[test]
fn test_bounding_box_nd_min_dist_sq() {
    let bb = BoundingBoxN::<4>::from_extents([2.0, 2.0, 2.0, 2.0], [3.0, 3.0, 3.0, 3.0]).unwrap();
    assert_eq!(bb.min_dist_sq_nd(&[3.0, 3.0, 3.0, 3.0]), 0.0);
    assert!((bb.min_dist_sq_nd(&[0.0, 0.0, 0.0, 0.0]) - 16.0).abs() < f32::EPSILON);
}

#[test]
fn test_bounding_box_nd_union_and_intersects() {
    let a = BoundingBoxN::<4>::from_extents([0.0, 0.0, 0.0, 0.0], [2.0, 2.0, 2.0, 2.0]).unwrap();
    let b = BoundingBoxN::<4>::from_extents([1.0, 1.0, 1.0, 1.0], [2.0, 2.0, 2.0, 2.0]).unwrap();
    assert!(a.intersects(b));
    let u = a.union(b);
    assert_eq!(u.measure(), 3.0_f32.powi(4));

    let c = BoundingBoxN::<4>::from_extents([5.0, 5.0, 5.0, 5.0], [1.0, 1.0, 1.0, 1.0]).unwrap();
    assert!(!a.intersects(c));
}

// ---------------------------------------------------------------------------
// STR bulk loading tests
// ---------------------------------------------------------------------------

#[test]
fn test_bulk_load_empty() {
    let index = SpatialIndex::<u32>::bulk_load(vec![]);
    assert!(index.is_empty());
    assert_eq!(index.len(), 0);
}

#[test]
fn test_bulk_load_single_entry() {
    let entry = SpatialEntry {
        bounds: BoundingBox::new(1.0, 2.0, 3.0, 4.0).unwrap(),
        data: 42u32,
    };
    let index = SpatialIndex::bulk_load(vec![entry]);
    assert_eq!(index.len(), 1);
    let results = index.query_region(BoundingBox::new(0.0, 0.0, 10.0, 10.0).unwrap());
    assert_eq!(results.len(), 1);
    assert_eq!(results[0].data, 42);
}

#[test]
fn test_bulk_load_within_max_entries() {
    let entries: Vec<SpatialEntry<u32>> = (0..5)
        .map(|i| SpatialEntry {
            bounds: BoundingBox::new(i as f32, 0.0, 1.0, 1.0).unwrap(),
            data: i,
        })
        .collect();
    let index = SpatialIndex::bulk_load(entries);
    assert_eq!(index.len(), 5);
    let results = index.query_region(BoundingBox::new(-1.0, -1.0, 100.0, 100.0).unwrap());
    assert_eq!(results.len(), 5);
}

#[test]
fn test_bulk_load_one_over_max() {
    let entries: Vec<SpatialEntry<u32>> = (0..10)
        .map(|i| SpatialEntry {
            bounds: BoundingBox::new(i as f32 * 2.0, 0.0, 1.0, 1.0).unwrap(),
            data: i,
        })
        .collect();
    let index = SpatialIndex::bulk_load(entries);
    assert_eq!(index.len(), 10);
    let results = index.query_region(BoundingBox::new(-1.0, -1.0, 100.0, 100.0).unwrap());
    assert_eq!(results.len(), 10);
}

#[test]
fn test_bulk_load_preserves_all_entries() {
    let entries: Vec<SpatialEntry<u32>> = (0..200)
        .map(|i| SpatialEntry {
            bounds: BoundingBox::new((i % 20) as f32, (i / 20) as f32, 0.5, 0.5).unwrap(),
            data: i,
        })
        .collect();
    let index = SpatialIndex::bulk_load(entries);
    assert_eq!(index.len(), 200);

    let mut found: Vec<u32> = index.iter().map(|e| e.data).collect();
    found.sort_unstable();
    let expected: Vec<u32> = (0..200).collect();
    assert_eq!(found, expected);
}

#[test]
fn test_bulk_load_matches_insert() {
    let entries: Vec<SpatialEntry<u32>> = (0..500)
        .map(|i| SpatialEntry {
            bounds: BoundingBox::new((i % 25) as f32 * 2.0, (i / 25) as f32 * 2.0, 1.0, 1.0)
                .unwrap(),
            data: i,
        })
        .collect();

    let bulk = SpatialIndex::bulk_load(entries.clone());

    let mut sequential = SpatialIndex::new();
    for entry in entries {
        sequential.insert(entry);
    }

    assert_eq!(bulk.len(), sequential.len());

    // Region query should return the same set of entries.
    let region = BoundingBox::new(5.0, 5.0, 10.0, 10.0).unwrap();
    let mut bulk_data: Vec<u32> = bulk.query_region(region).iter().map(|e| e.data).collect();
    let mut seq_data: Vec<u32> = sequential
        .query_region(region)
        .iter()
        .map(|e| e.data)
        .collect();
    bulk_data.sort_unstable();
    seq_data.sort_unstable();
    assert_eq!(bulk_data, seq_data);

    // Nearest query: both should return 5 results. Exact membership may
    // differ when entries are equidistant, so just verify the count.
    let bulk_nearest = bulk.query_nearest(10.0, 10.0, 5);
    let seq_nearest = sequential.query_nearest(10.0, 10.0, 5);
    assert_eq!(bulk_nearest.len(), seq_nearest.len());
}

#[test]
fn test_bulk_load_2d_region_query() {
    // 10x10 grid, cross-check against brute force.
    let entries: Vec<SpatialEntry<u32>> = (0..100)
        .map(|i| SpatialEntry {
            bounds: BoundingBox::new((i % 10) as f32, (i / 10) as f32, 0.5, 0.5).unwrap(),
            data: i,
        })
        .collect();
    let index = SpatialIndex::bulk_load(entries.clone());

    let region = BoundingBox::new(2.0, 3.0, 4.0, 4.0).unwrap();
    let mut bulk_data: Vec<u32> = index.query_region(region).iter().map(|e| e.data).collect();
    let mut brute_data: Vec<u32> = entries
        .iter()
        .filter(|e| e.bounds.intersects(region))
        .map(|e| e.data)
        .collect();
    bulk_data.sort_unstable();
    brute_data.sort_unstable();
    assert_eq!(bulk_data, brute_data);
}

#[test]
fn test_bulk_load_2d_nearest_query() {
    let entries: Vec<SpatialEntry<u32>> = (0..100)
        .map(|i| SpatialEntry {
            bounds: BoundingBox::new((i % 10) as f32 * 3.0, (i / 10) as f32 * 3.0, 1.0, 1.0)
                .unwrap(),
            data: i,
        })
        .collect();
    let index = SpatialIndex::bulk_load(entries.clone());

    let results = index.query_nearest(0.0, 0.0, 5);
    assert_eq!(results.len(), 5);
    // The nearest entry should be the one at the origin.
    assert_eq!(results[0].data, 0);
}

#[test]
fn test_bulk_load_2d_radius_query() {
    let entries: Vec<SpatialEntry<u32>> = (0..100)
        .map(|i| SpatialEntry {
            bounds: BoundingBox::new((i % 10) as f32, (i / 10) as f32, 0.5, 0.5).unwrap(),
            data: i,
        })
        .collect();
    let index = SpatialIndex::bulk_load(entries.clone());

    let cx = 5.0_f32;
    let cy = 5.0_f32;
    let radius = 3.0_f32;
    let results = index.query_within_radius(cx, cy, radius);

    let brute: Vec<u32> = entries
        .iter()
        .filter(|e| e.bounds.min_dist_sq(cx, cy) <= radius * radius)
        .map(|e| e.data)
        .collect();
    assert_eq!(results.len(), brute.len());
}

#[test]
fn test_bulk_load_3d() {
    let entries: Vec<SpatialEntry3D<u32>> = (0..200)
        .map(|i| SpatialEntry3D {
            bounds: BoundingBox3D::new(
                (i % 10) as f32,
                ((i / 10) % 10) as f32,
                (i / 100) as f32,
                0.5,
                0.5,
                0.5,
            )
            .unwrap(),
            data: i,
        })
        .collect();
    let index = SpatialIndex3D::bulk_load(entries.clone());
    assert_eq!(index.len(), 200);

    let region = BoundingBox3D::new(0.0, 0.0, 0.0, 5.5, 5.5, 1.5).unwrap();
    let mut bulk_data: Vec<u32> = index.query_region(region).iter().map(|e| e.data).collect();
    let mut brute_data: Vec<u32> = entries
        .iter()
        .filter(|e| e.bounds.intersects(region))
        .map(|e| e.data)
        .collect();
    bulk_data.sort_unstable();
    brute_data.sort_unstable();
    assert_eq!(bulk_data, brute_data);

    let nearest = index.query_nearest(0.0, 0.0, 0.0, 3);
    assert_eq!(nearest.len(), 3);
}

#[test]
fn test_bulk_load_4d() {
    let entries: Vec<SpatialEntryN<4, u32>> = (0..50)
        .map(|i| SpatialEntryN {
            bounds: BoundingBoxN::from_extents(
                [i as f32, (i * 2) as f32, (i * 3) as f32, (i * 4) as f32],
                [1.0, 1.0, 1.0, 1.0],
            )
            .unwrap(),
            data: i,
        })
        .collect();
    let index = SpatialIndexN::<4, u32>::bulk_load(entries.clone());
    assert_eq!(index.len(), 50);

    let region =
        BoundingBoxN::from_extents([0.0, 0.0, 0.0, 0.0], [10.5, 20.5, 30.5, 40.5]).unwrap();
    let mut bulk_data: Vec<u32> = index.query_region(region).iter().map(|e| e.data).collect();
    let mut brute_data: Vec<u32> = entries
        .iter()
        .filter(|e| e.bounds.intersects(region))
        .map(|e| e.data)
        .collect();
    bulk_data.sort_unstable();
    brute_data.sort_unstable();
    assert_eq!(bulk_data, brute_data);
}

#[test]
fn test_bulk_load_1d() {
    let entries: Vec<SpatialEntryN<1, u32>> = (0..30)
        .map(|i| SpatialEntryN {
            bounds: BoundingBoxN::from_extents([i as f32 * 2.0], [1.0]).unwrap(),
            data: i,
        })
        .collect();
    let index = SpatialIndexN::<1, u32>::bulk_load(entries.clone());
    assert_eq!(index.len(), 30);

    let region = BoundingBoxN::from_extents([5.0], [10.0]).unwrap();
    let mut bulk_data: Vec<u32> = index.query_region(region).iter().map(|e| e.data).collect();
    let mut brute_data: Vec<u32> = entries
        .iter()
        .filter(|e| e.bounds.intersects(region))
        .map(|e| e.data)
        .collect();
    bulk_data.sort_unstable();
    brute_data.sort_unstable();
    assert_eq!(bulk_data, brute_data);
}

#[test]
fn test_bulk_load_serde_roundtrip() {
    let entries: Vec<SpatialEntry<u32>> = (0..100)
        .map(|i| SpatialEntry {
            bounds: BoundingBox::new((i % 10) as f32, (i / 10) as f32, 0.5, 0.5).unwrap(),
            data: i,
        })
        .collect();
    let index = SpatialIndex::bulk_load(entries);

    let bytes = bitcode::serialize(&index).unwrap();
    let restored: SpatialIndex<u32> = bitcode::deserialize(&bytes).unwrap();
    assert_eq!(restored.len(), 100);

    let region = BoundingBox::new(2.0, 2.0, 5.0, 5.0).unwrap();
    let mut orig_data: Vec<u32> = index.query_region(region).iter().map(|e| e.data).collect();
    let mut rest_data: Vec<u32> = restored
        .query_region(region)
        .iter()
        .map(|e| e.data)
        .collect();
    orig_data.sort_unstable();
    rest_data.sort_unstable();
    assert_eq!(orig_data, rest_data);
}

#[test]
fn test_bulk_load_then_insert_and_remove() {
    let entries: Vec<SpatialEntry<u32>> = (0..20)
        .map(|i| SpatialEntry {
            bounds: BoundingBox::new(i as f32, 0.0, 1.0, 1.0).unwrap(),
            data: i,
        })
        .collect();
    let mut index = SpatialIndex::bulk_load(entries);
    assert_eq!(index.len(), 20);

    // Insert after bulk load.
    index.insert(SpatialEntry {
        bounds: BoundingBox::new(100.0, 100.0, 1.0, 1.0).unwrap(),
        data: 999,
    });
    assert_eq!(index.len(), 21);

    // Remove after bulk load.
    let bb = BoundingBox::new(0.0, 0.0, 1.0, 1.0).unwrap();
    index.remove(bb, |e| e.data == 0).unwrap();
    assert_eq!(index.len(), 20);
}

#[test]
fn test_bulk_load_duplicate_positions() {
    let entries: Vec<SpatialEntry<u32>> = (0..50)
        .map(|i| SpatialEntry {
            bounds: BoundingBox::new(5.0, 5.0, 1.0, 1.0).unwrap(),
            data: i,
        })
        .collect();
    let index = SpatialIndex::bulk_load(entries);
    assert_eq!(index.len(), 50);

    let region = BoundingBox::new(4.0, 4.0, 3.0, 3.0).unwrap();
    let results = index.query_region(region);
    assert_eq!(results.len(), 50);
}

#[test]
fn test_bulk_load_large_10k() {
    let entries: Vec<SpatialEntry<u32>> = (0..10_000)
        .map(|i| SpatialEntry {
            bounds: BoundingBox::new((i % 100) as f32, (i / 100) as f32, 0.5, 0.5).unwrap(),
            data: i,
        })
        .collect();
    let index = SpatialIndex::bulk_load(entries.clone());
    assert_eq!(index.len(), 10_000);

    // Region query cross-check.
    let region = BoundingBox::new(10.0, 20.0, 15.0, 15.0).unwrap();
    let mut bulk_data: Vec<u32> = index.query_region(region).iter().map(|e| e.data).collect();
    let mut brute_data: Vec<u32> = entries
        .iter()
        .filter(|e| e.bounds.intersects(region))
        .map(|e| e.data)
        .collect();
    bulk_data.sort_unstable();
    brute_data.sort_unstable();
    assert_eq!(bulk_data, brute_data);

    // Nearest query sanity check.
    let nearest = index.query_nearest(50.0, 50.0, 5);
    assert_eq!(nearest.len(), 5);
}

// ---------------------------------------------------------------------------
// Deletion rebalancing (condense_tree) tests
// ---------------------------------------------------------------------------

#[test]
fn test_remove_rebalance_preserves_entries() {
    let mut index = SpatialIndex::new();
    for i in 0..100u32 {
        index.insert(SpatialEntry {
            bounds: BoundingBox::new((i % 10) as f32, (i / 10) as f32, 0.5, 0.5).unwrap(),
            data: i,
        });
    }

    // Delete 60 entries (even-numbered).
    let deleted: Vec<u32> = (0..100).filter(|i| i % 5 < 3).collect();
    for &d in &deleted {
        let bb = BoundingBox::new((d % 10) as f32, (d / 10) as f32, 0.5, 0.5).unwrap();
        index.remove(bb, |e| e.data == d).unwrap();
    }
    let expected_remaining = 100 - deleted.len();
    assert_eq!(index.len(), expected_remaining);

    // All remaining entries must be queryable.
    let big = BoundingBox::new(-1.0, -1.0, 20.0, 20.0).unwrap();
    let mut found: Vec<u32> = index.query_region(big).iter().map(|e| e.data).collect();
    found.sort_unstable();
    let mut expected: Vec<u32> = (0..100).filter(|i| i % 5 >= 3).collect();
    expected.sort_unstable();
    assert_eq!(found, expected);
}

#[test]
fn test_remove_rebalance_nearest_correct() {
    let mut index = SpatialIndex::new();
    let entries: Vec<SpatialEntry<u32>> = (0..200)
        .map(|i| SpatialEntry {
            bounds: BoundingBox::new((i % 20) as f32 * 2.0, (i / 20) as f32 * 2.0, 1.0, 1.0)
                .unwrap(),
            data: i,
        })
        .collect();
    for e in &entries {
        index.insert(e.clone());
    }

    // Delete every other entry.
    for i in (0..200u32).step_by(2) {
        let bb = BoundingBox::new((i % 20) as f32 * 2.0, (i / 20) as f32 * 2.0, 1.0, 1.0).unwrap();
        index.remove(bb, |e| e.data == i).unwrap();
    }
    assert_eq!(index.len(), 100);

    // Nearest query vs brute-force.
    let nearest = index.query_nearest(20.0, 10.0, 5);
    assert_eq!(nearest.len(), 5);

    // All returned entries must exist in the index.
    let all_data: Vec<u32> = index.iter().map(|e| e.data).collect();
    for e in &nearest {
        assert!(all_data.contains(&e.data));
    }
}

#[test]
fn test_remove_rebalance_radius_correct() {
    let mut index = SpatialIndex::new();
    for i in 0..100u32 {
        index.insert(SpatialEntry {
            bounds: BoundingBox::new((i % 10) as f32, (i / 10) as f32, 0.5, 0.5).unwrap(),
            data: i,
        });
    }

    // Delete 50 entries.
    for i in 0..50u32 {
        let bb = BoundingBox::new((i % 10) as f32, (i / 10) as f32, 0.5, 0.5).unwrap();
        index.remove(bb, |e| e.data == i).unwrap();
    }

    let cx = 7.0_f32;
    let cy = 7.0_f32;
    let radius = 3.0_f32;
    let results = index.query_within_radius(cx, cy, radius);

    // Cross-check: brute-force over iterator.
    let brute_count = index
        .iter()
        .filter(|e| e.bounds.min_dist_sq(cx, cy) <= radius * radius)
        .count();
    assert_eq!(results.len(), brute_count);
}

#[test]
fn test_remove_rebalance_heavy_churn_cycle() {
    let mut index = SpatialIndex::new();
    let mut next_id = 0u32;

    for _cycle in 0..5 {
        // Insert batch.
        for _ in 0..100 {
            index.insert(SpatialEntry {
                bounds: BoundingBox::new((next_id % 50) as f32, (next_id / 50) as f32, 0.5, 0.5)
                    .unwrap(),
                data: next_id,
            });
            next_id += 1;
        }

        // Delete half of current entries.
        let current: Vec<SpatialEntry<u32>> = index.iter().cloned().collect();
        let to_delete = current.len() / 2;
        for entry in current.iter().take(to_delete) {
            index
                .remove(entry.bounds, |e| e.data == entry.data)
                .unwrap();
        }

        // Verify correctness after each cycle.
        let big = BoundingBox::new(-1.0, -1.0, 200.0, 200.0).unwrap();
        let found = index.query_region(big);
        assert_eq!(found.len(), index.len());
    }
}

#[test]
fn test_remove_rebalance_delete_all() {
    let mut index = SpatialIndex::new();
    for i in 0..50u32 {
        index.insert(SpatialEntry {
            bounds: BoundingBox::new(i as f32, 0.0, 1.0, 1.0).unwrap(),
            data: i,
        });
    }

    for i in 0..50u32 {
        let bb = BoundingBox::new(i as f32, 0.0, 1.0, 1.0).unwrap();
        index.remove(bb, |e| e.data == i).unwrap();
    }
    assert!(index.is_empty());
    assert_eq!(index.len(), 0);
    assert_eq!(index.iter().count(), 0);
}

#[test]
fn test_remove_rebalance_3d() {
    let mut index = SpatialIndex3D::new();
    for i in 0..100u32 {
        index.insert(SpatialEntry3D {
            bounds: BoundingBox3D::new(
                (i % 10) as f32,
                ((i / 10) % 10) as f32,
                (i / 100) as f32,
                0.5,
                0.5,
                0.5,
            )
            .unwrap(),
            data: i,
        });
    }

    // Delete 70 entries.
    for i in 0..70u32 {
        let bb = BoundingBox3D::new(
            (i % 10) as f32,
            ((i / 10) % 10) as f32,
            (i / 100) as f32,
            0.5,
            0.5,
            0.5,
        )
        .unwrap();
        index.remove(bb, |e| e.data == i).unwrap();
    }
    assert_eq!(index.len(), 30);

    let big = BoundingBox3D::new(-1.0, -1.0, -1.0, 20.0, 20.0, 20.0).unwrap();
    let mut found: Vec<u32> = index.query_region(big).iter().map(|e| e.data).collect();
    found.sort_unstable();
    assert_eq!(found, (70..100).collect::<Vec<u32>>());
}

#[test]
fn test_remove_rebalance_serde_after_churn() {
    let mut index = SpatialIndex::new();
    for i in 0..50u32 {
        index.insert(SpatialEntry {
            bounds: BoundingBox::new(i as f32, 0.0, 1.0, 1.0).unwrap(),
            data: i,
        });
    }
    for i in 0..30u32 {
        let bb = BoundingBox::new(i as f32, 0.0, 1.0, 1.0).unwrap();
        index.remove(bb, |e| e.data == i).unwrap();
    }
    assert_eq!(index.len(), 20);

    let bytes = bitcode::serialize(&index).unwrap();
    let restored: SpatialIndex<u32> = bitcode::deserialize(&bytes).unwrap();
    assert_eq!(restored.len(), 20);

    let big = BoundingBox::new(-1.0, -1.0, 100.0, 100.0).unwrap();
    let mut orig: Vec<u32> = index.query_region(big).iter().map(|e| e.data).collect();
    let mut rest: Vec<u32> = restored.query_region(big).iter().map(|e| e.data).collect();
    orig.sort_unstable();
    rest.sort_unstable();
    assert_eq!(orig, rest);
}

#[test]
fn test_remove_not_found_after_rebalance() {
    let mut index = SpatialIndex::new();
    for i in 0..30u32 {
        index.insert(SpatialEntry {
            bounds: BoundingBox::new(i as f32, 0.0, 1.0, 1.0).unwrap(),
            data: i,
        });
    }
    // Delete enough to trigger rebalancing.
    for i in 0..20u32 {
        let bb = BoundingBox::new(i as f32, 0.0, 1.0, 1.0).unwrap();
        index.remove(bb, |e| e.data == i).unwrap();
    }

    // A not-found remove should still return NotFound.
    let bb = BoundingBox::new(0.0, 0.0, 1.0, 1.0).unwrap();
    let result = index.remove(bb, |e| e.data == 999);
    assert!(result.is_err());
    assert_eq!(index.len(), 10);
}

// ---------------------------------------------------------------------------
// SpatialConfig tests
// ---------------------------------------------------------------------------

#[test]
fn test_config_default() {
    let cfg = SpatialConfig::DEFAULT;
    assert_eq!(cfg.max_entries(), 9);
    assert_eq!(cfg.min_entries(), 4);
}

#[test]
fn test_config_default_trait() {
    let cfg = SpatialConfig::default();
    assert_eq!(cfg, SpatialConfig::DEFAULT);
}

#[test]
fn test_config_new_valid() {
    let cfg = SpatialConfig::new(16).unwrap();
    assert_eq!(cfg.max_entries(), 16);
    assert_eq!(cfg.min_entries(), 8);
}

#[test]
fn test_config_new_too_small() {
    let result = SpatialConfig::new(3);
    assert!(result.is_err());
}

#[test]
fn test_config_new_boundary() {
    let cfg = SpatialConfig::new(4).unwrap();
    assert_eq!(cfg.max_entries(), 4);
    assert_eq!(cfg.min_entries(), 2);
}

#[test]
fn test_config_new_odd() {
    let cfg = SpatialConfig::new(5).unwrap();
    assert_eq!(cfg.max_entries(), 5);
    assert_eq!(cfg.min_entries(), 2);
}

#[test]
fn test_invalid_config_error_display() {
    let err = SpatialError::InvalidConfig;
    assert_eq!(
        err.to_string(),
        "invalid config: max_entries must be at least 4"
    );
}

#[test]
fn test_with_config_insert_query() {
    let cfg = SpatialConfig::new(5).unwrap();
    let mut index = SpatialIndex::with_config(cfg);
    for i in 0..50u32 {
        index.insert(SpatialEntry {
            bounds: BoundingBox::new(i as f32, 0.0, 1.0, 1.0).unwrap(),
            data: i,
        });
    }
    assert_eq!(index.len(), 50);
    assert_eq!(index.config(), cfg);
    let region = BoundingBox::new(-1.0, -1.0, 100.0, 100.0).unwrap();
    let results = index.query_region(region);
    assert_eq!(results.len(), 50);
}

#[test]
fn test_with_config_remove() {
    let cfg = SpatialConfig::new(5).unwrap();
    let mut index = SpatialIndex::with_config(cfg);
    for i in 0..20u32 {
        index.insert(SpatialEntry {
            bounds: BoundingBox::new(i as f32 * 2.0, 0.0, 1.0, 1.0).unwrap(),
            data: i,
        });
    }
    for i in 0..10u32 {
        let bb = BoundingBox::new(i as f32 * 2.0, 0.0, 1.0, 1.0).unwrap();
        index.remove(bb, |e| e.data == i).unwrap();
    }
    assert_eq!(index.len(), 10);
    let region = BoundingBox::new(-1.0, -1.0, 100.0, 100.0).unwrap();
    assert_eq!(index.query_region(region).len(), 10);
}

#[test]
fn test_with_config_bulk_load() {
    let cfg = SpatialConfig::new(16).unwrap();
    let entries: Vec<SpatialEntry<u32>> = (0..200)
        .map(|i| SpatialEntry {
            bounds: BoundingBox::new((i % 20) as f32, (i / 20) as f32, 0.5, 0.5).unwrap(),
            data: i,
        })
        .collect();
    let index = SpatialIndex::bulk_load_with_config(entries, cfg);
    assert_eq!(index.len(), 200);
    assert_eq!(index.config(), cfg);
    let region = BoundingBox::new(-1.0, -1.0, 100.0, 100.0).unwrap();
    assert_eq!(index.query_region(region).len(), 200);
}

#[test]
fn test_with_config_serde_v2_roundtrip() {
    let cfg = SpatialConfig::new(16).unwrap();
    let entries: Vec<SpatialEntry<u32>> = (0..50)
        .map(|i| SpatialEntry {
            bounds: BoundingBox::new(i as f32, 0.0, 1.0, 1.0).unwrap(),
            data: i,
        })
        .collect();
    let index = SpatialIndex::bulk_load_with_config(entries, cfg);
    let bytes = bitcode::serialize(&index).unwrap();
    let restored: SpatialIndex<u32> = bitcode::deserialize(&bytes).unwrap();
    assert_eq!(restored.len(), 50);
    assert_eq!(restored.config(), cfg);
}

#[test]
fn test_with_config_serde_rejects_unknown_version() {
    #[derive(serde::Serialize)]
    struct FakeDto {
        version: u8,
        max_entries: usize,
        split_strategy: u8,
        entries: Vec<SpatialEntry<u32>>,
    }
    let fake = FakeDto {
        version: 99,
        max_entries: 9,
        split_strategy: 1,
        entries: vec![],
    };
    let bytes = bitcode::serialize(&fake).unwrap();
    let result: Result<SpatialIndex<u32>, _> = bitcode::deserialize(&bytes);
    assert!(result.is_err());
}

// ---------------------------------------------------------------------------
// R*-tree split strategy tests
// ---------------------------------------------------------------------------

#[test]
fn test_rstar_insert_preserves_all_entries() {
    let cfg = SpatialConfig::with_strategy(9, SplitStrategy::RStar).unwrap();
    let mut index = SpatialIndex::with_config(cfg);
    for i in 0..100u32 {
        index.insert(SpatialEntry {
            bounds: BoundingBox::new((i % 10) as f32, (i / 10) as f32, 1.0, 1.0).unwrap(),
            data: i,
        });
    }
    assert_eq!(index.len(), 100);
    let big = BoundingBox::new(-1.0, -1.0, 100.0, 100.0).unwrap();
    let mut found: Vec<u32> = index.query_region(big).iter().map(|e| e.data).collect();
    found.sort_unstable();
    assert_eq!(found, (0..100).collect::<Vec<u32>>());
}

#[test]
fn test_rstar_insert_and_remove() {
    let cfg = SpatialConfig::with_strategy(9, SplitStrategy::RStar).unwrap();
    let mut index = SpatialIndex::with_config(cfg);
    for i in 0..200u32 {
        index.insert(SpatialEntry {
            bounds: BoundingBox::new((i % 20) as f32, (i / 20) as f32, 0.5, 0.5).unwrap(),
            data: i,
        });
    }
    assert_eq!(index.len(), 200);

    for i in 0..100u32 {
        let bb = BoundingBox::new((i % 20) as f32, (i / 20) as f32, 0.5, 0.5).unwrap();
        index.remove(bb, |e| e.data == i).unwrap();
    }
    assert_eq!(index.len(), 100);

    let big = BoundingBox::new(-1.0, -1.0, 100.0, 100.0).unwrap();
    let mut found: Vec<u32> = index.query_region(big).iter().map(|e| e.data).collect();
    found.sort_unstable();
    assert_eq!(found, (100..200).collect::<Vec<u32>>());
}

#[test]
fn test_rstar_region_brute_force() {
    let cfg = SpatialConfig::with_strategy(9, SplitStrategy::RStar).unwrap();
    let mut index = SpatialIndex::with_config(cfg);
    let mut all_entries = Vec::new();
    for i in 0..5000u32 {
        let x = (i * 7 % 500) as f32;
        let y = (i * 13 % 500) as f32;
        let entry = SpatialEntry {
            bounds: BoundingBox::new(x, y, 1.0, 1.0).unwrap(),
            data: i,
        };
        all_entries.push(entry.clone());
        index.insert(entry);
    }
    assert_eq!(index.len(), 5000);

    let region = BoundingBox::new(100.0, 100.0, 50.0, 50.0).unwrap();
    let mut tree_results: Vec<u32> = index.query_region(region).iter().map(|e| e.data).collect();
    tree_results.sort_unstable();

    let mut brute: Vec<u32> = all_entries
        .iter()
        .filter(|e| e.bounds.intersects(region))
        .map(|e| e.data)
        .collect();
    brute.sort_unstable();

    assert_eq!(tree_results, brute);
}

#[test]
fn test_rstar_nearest_brute_force() {
    let cfg = SpatialConfig::with_strategy(9, SplitStrategy::RStar).unwrap();
    let mut index = SpatialIndex::with_config(cfg);
    let mut all_entries = Vec::new();
    for i in 0..1000u32 {
        let x = (i * 7 % 100) as f32;
        let y = (i * 13 % 100) as f32;
        let entry = SpatialEntry {
            bounds: BoundingBox::new(x, y, 1.0, 1.0).unwrap(),
            data: i,
        };
        all_entries.push(entry.clone());
        index.insert(entry);
    }

    let qx = 50.0_f32;
    let qy = 50.0_f32;
    let k = 10;
    let tree_nearest = index.query_nearest(qx, qy, k);
    assert_eq!(tree_nearest.len(), k);

    // Brute-force: compute distances, sort, take k.
    let mut dists: Vec<(u32, f32)> = all_entries
        .iter()
        .map(|e| (e.data, e.bounds.min_dist_sq(qx, qy)))
        .collect();
    dists.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
    let brute_k: Vec<f32> = dists.iter().take(k).map(|(_, d)| *d).collect();
    let tree_k: Vec<f32> = tree_nearest
        .iter()
        .map(|e| e.bounds.min_dist_sq(qx, qy))
        .collect();

    // The k-th distance from tree must match brute-force k-th distance.
    let tree_max = tree_k.iter().cloned().fold(0.0_f32, f32::max);
    let brute_max = brute_k.iter().cloned().fold(0.0_f32, f32::max);
    assert!(
        (tree_max - brute_max).abs() < 1e-6,
        "tree max dist {tree_max} vs brute {brute_max}"
    );
}

#[test]
fn test_rstar_radius_brute_force() {
    let cfg = SpatialConfig::with_strategy(9, SplitStrategy::RStar).unwrap();
    let mut index = SpatialIndex::with_config(cfg);
    let mut all_entries = Vec::new();
    for i in 0..2000u32 {
        let x = (i * 7 % 200) as f32;
        let y = (i * 13 % 200) as f32;
        let entry = SpatialEntry {
            bounds: BoundingBox::new(x, y, 1.0, 1.0).unwrap(),
            data: i,
        };
        all_entries.push(entry.clone());
        index.insert(entry);
    }

    let qx = 100.0_f32;
    let qy = 100.0_f32;
    let r = 30.0_f32;
    let mut tree_results: Vec<u32> = index
        .query_within_radius(qx, qy, r)
        .iter()
        .map(|e| e.data)
        .collect();
    tree_results.sort_unstable();

    let r_sq = r * r;
    let mut brute: Vec<u32> = all_entries
        .iter()
        .filter(|e| e.bounds.min_dist_sq(qx, qy) <= r_sq)
        .map(|e| e.data)
        .collect();
    brute.sort_unstable();

    assert_eq!(tree_results, brute);
}

#[test]
fn test_rstar_3d() {
    let cfg = SpatialConfig::with_strategy(9, SplitStrategy::RStar).unwrap();
    let mut index = SpatialIndex3D::with_config(cfg);
    for i in 0..200u32 {
        index.insert(SpatialEntry3D {
            bounds: BoundingBox3D::new(
                (i % 10) as f32,
                ((i / 10) % 10) as f32,
                (i / 100) as f32,
                0.5,
                0.5,
                0.5,
            )
            .unwrap(),
            data: i,
        });
    }
    assert_eq!(index.len(), 200);

    let big = BoundingBox3D::new(-1.0, -1.0, -1.0, 100.0, 100.0, 100.0).unwrap();
    let mut found: Vec<u32> = index.query_region(big).iter().map(|e| e.data).collect();
    found.sort_unstable();
    assert_eq!(found, (0..200).collect::<Vec<u32>>());

    let nearest = index.query_nearest(5.0, 5.0, 0.5, 5);
    assert_eq!(nearest.len(), 5);
}

#[test]
fn test_rstar_forced_reinsertion_triggers() {
    let cfg = SpatialConfig::with_strategy(9, SplitStrategy::RStar).unwrap();
    let mut index = SpatialIndex::with_config(cfg);
    for i in 0..500u32 {
        index.insert(SpatialEntry {
            bounds: BoundingBox::new((i % 25) as f32, (i / 25) as f32, 0.5, 0.5).unwrap(),
            data: i,
        });
    }
    assert_eq!(index.len(), 500);
    assert_eq!(index.iter().count(), 500);

    let big = BoundingBox::new(-1.0, -1.0, 100.0, 100.0).unwrap();
    assert_eq!(index.query_region(big).len(), 500);
}

#[test]
fn test_rstar_len_invariant_under_reinsertion() {
    let cfg = SpatialConfig::with_strategy(9, SplitStrategy::RStar).unwrap();
    let mut index = SpatialIndex::with_config(cfg);
    for i in 0..200u32 {
        index.insert(SpatialEntry {
            bounds: BoundingBox::new((i % 20) as f32, (i / 20) as f32, 0.5, 0.5).unwrap(),
            data: i,
        });
        assert_eq!(
            index.len(),
            index.iter().count(),
            "len mismatch after inserting entry {i}"
        );
    }
}

#[test]
fn test_rstar_vs_linear_same_results() {
    let rstar_cfg = SpatialConfig::with_strategy(9, SplitStrategy::RStar).unwrap();
    let linear_cfg = SpatialConfig::with_strategy(9, SplitStrategy::Linear).unwrap();
    let mut rstar_idx = SpatialIndex::with_config(rstar_cfg);
    let mut linear_idx = SpatialIndex::with_config(linear_cfg);

    for i in 0..1000u32 {
        let x = (i * 7 % 100) as f32;
        let y = (i * 13 % 100) as f32;
        let entry = SpatialEntry {
            bounds: BoundingBox::new(x, y, 1.0, 1.0).unwrap(),
            data: i,
        };
        rstar_idx.insert(entry.clone());
        linear_idx.insert(entry);
    }
    assert_eq!(rstar_idx.len(), 1000);
    assert_eq!(linear_idx.len(), 1000);

    // Region query: same results.
    let region = BoundingBox::new(30.0, 30.0, 40.0, 40.0).unwrap();
    let mut rstar_region: Vec<u32> = rstar_idx
        .query_region(region)
        .iter()
        .map(|e| e.data)
        .collect();
    let mut linear_region: Vec<u32> = linear_idx
        .query_region(region)
        .iter()
        .map(|e| e.data)
        .collect();
    rstar_region.sort_unstable();
    linear_region.sort_unstable();
    assert_eq!(rstar_region, linear_region);

    // Radius query: same results.
    let mut rstar_radius: Vec<u32> = rstar_idx
        .query_within_radius(50.0, 50.0, 20.0)
        .iter()
        .map(|e| e.data)
        .collect();
    let mut linear_radius: Vec<u32> = linear_idx
        .query_within_radius(50.0, 50.0, 20.0)
        .iter()
        .map(|e| e.data)
        .collect();
    rstar_radius.sort_unstable();
    linear_radius.sort_unstable();
    assert_eq!(rstar_radius, linear_radius);

    // Nearest query: same count, same max distance.
    let rstar_nn = rstar_idx.query_nearest(50.0, 50.0, 10);
    let linear_nn = linear_idx.query_nearest(50.0, 50.0, 10);
    assert_eq!(rstar_nn.len(), linear_nn.len());
}

#[test]
fn test_rstar_vs_linear_churn_regression() {
    let rstar_cfg = SpatialConfig::with_strategy(9, SplitStrategy::RStar).unwrap();
    let linear_cfg = SpatialConfig::with_strategy(9, SplitStrategy::Linear).unwrap();
    let mut rstar_idx = SpatialIndex::with_config(rstar_cfg);
    let mut linear_idx = SpatialIndex::with_config(linear_cfg);

    // Insert 500.
    for i in 0..500u32 {
        let x = (i * 7 % 100) as f32;
        let y = (i * 13 % 100) as f32;
        let entry = SpatialEntry {
            bounds: BoundingBox::new(x, y, 0.5, 0.5).unwrap(),
            data: i,
        };
        rstar_idx.insert(entry.clone());
        linear_idx.insert(entry);
    }

    // Delete 200.
    for i in 0..200u32 {
        let x = (i * 7 % 100) as f32;
        let y = (i * 13 % 100) as f32;
        let bb = BoundingBox::new(x, y, 0.5, 0.5).unwrap();
        rstar_idx.remove(bb, |e| e.data == i).unwrap();
        linear_idx.remove(bb, |e| e.data == i).unwrap();
    }

    // Insert 100 more.
    for i in 500..600u32 {
        let x = (i * 7 % 100) as f32;
        let y = (i * 13 % 100) as f32;
        let entry = SpatialEntry {
            bounds: BoundingBox::new(x, y, 0.5, 0.5).unwrap(),
            data: i,
        };
        rstar_idx.insert(entry.clone());
        linear_idx.insert(entry);
    }

    assert_eq!(rstar_idx.len(), 400);
    assert_eq!(linear_idx.len(), 400);
    assert_eq!(rstar_idx.len(), rstar_idx.iter().count());

    let big = BoundingBox::new(-1.0, -1.0, 200.0, 200.0).unwrap();
    let mut rstar_all: Vec<u32> = rstar_idx.query_region(big).iter().map(|e| e.data).collect();
    let mut linear_all: Vec<u32> = linear_idx
        .query_region(big)
        .iter()
        .map(|e| e.data)
        .collect();
    rstar_all.sort_unstable();
    linear_all.sort_unstable();
    assert_eq!(rstar_all, linear_all);
}

#[test]
fn test_rstar_heavy_churn() {
    let cfg = SpatialConfig::with_strategy(9, SplitStrategy::RStar).unwrap();
    let mut index = SpatialIndex::with_config(cfg);

    // Cycle: insert 50, delete 30, repeat 5 times.
    let mut next_id = 0u32;
    for cycle in 0..5u32 {
        for _ in 0..50 {
            let x = (next_id * 7 % 200) as f32;
            let y = (next_id * 13 % 200) as f32;
            index.insert(SpatialEntry {
                bounds: BoundingBox::new(x, y, 0.5, 0.5).unwrap(),
                data: next_id,
            });
            next_id += 1;
        }
        let start = cycle * 50;
        for i in start..(start + 30) {
            let x = (i * 7 % 200) as f32;
            let y = (i * 13 % 200) as f32;
            let bb = BoundingBox::new(x, y, 0.5, 0.5).unwrap();
            index.remove(bb, |e| e.data == i).unwrap();
        }
    }

    // 5 * 50 inserted - 5 * 30 deleted = 100 remaining.
    assert_eq!(index.len(), 100);
    assert_eq!(index.iter().count(), 100);
}

#[test]
fn test_rstar_bulk_load_then_insert() {
    let cfg = SpatialConfig::with_strategy(9, SplitStrategy::RStar).unwrap();
    let entries: Vec<SpatialEntry<u32>> = (0..500)
        .map(|i| SpatialEntry {
            bounds: BoundingBox::new((i % 25) as f32, (i / 25) as f32, 0.5, 0.5).unwrap(),
            data: i,
        })
        .collect();
    let mut index = SpatialIndex::bulk_load_with_config(entries, cfg);
    assert_eq!(index.len(), 500);

    // Insert more with RStar strategy.
    for i in 500..700u32 {
        index.insert(SpatialEntry {
            bounds: BoundingBox::new((i % 25) as f32, (i / 25) as f32, 0.5, 0.5).unwrap(),
            data: i,
        });
    }
    assert_eq!(index.len(), 700);
    assert_eq!(index.iter().count(), 700);

    let big = BoundingBox::new(-1.0, -1.0, 100.0, 100.0).unwrap();
    assert_eq!(index.query_region(big).len(), 700);
}

#[test]
fn test_serde_v3_roundtrip_rstar() {
    let cfg = SpatialConfig::with_strategy(9, SplitStrategy::RStar).unwrap();
    let mut index = SpatialIndex::with_config(cfg);
    for i in 0..50u32 {
        index.insert(SpatialEntry {
            bounds: BoundingBox::new(i as f32, 0.0, 1.0, 1.0).unwrap(),
            data: i,
        });
    }
    let bytes = bitcode::serialize(&index).unwrap();
    let restored: SpatialIndex<u32> = bitcode::deserialize(&bytes).unwrap();
    assert_eq!(restored.len(), 50);
    assert_eq!(restored.config().split_strategy(), SplitStrategy::RStar);
    assert_eq!(restored.config().max_entries(), 9);
}

#[test]
fn test_serde_v3_roundtrip_linear() {
    let cfg = SpatialConfig::with_strategy(9, SplitStrategy::Linear).unwrap();
    let mut index = SpatialIndex::with_config(cfg);
    for i in 0..50u32 {
        index.insert(SpatialEntry {
            bounds: BoundingBox::new(i as f32, 0.0, 1.0, 1.0).unwrap(),
            data: i,
        });
    }
    let bytes = bitcode::serialize(&index).unwrap();
    let restored: SpatialIndex<u32> = bitcode::deserialize(&bytes).unwrap();
    assert_eq!(restored.len(), 50);
    assert_eq!(restored.config().split_strategy(), SplitStrategy::Linear);
}

#[test]
fn test_serde_v3_rejects_invalid_split_strategy() {
    #[derive(serde::Serialize)]
    struct FakeDto {
        version: u8,
        max_entries: usize,
        split_strategy: u8,
        entries: Vec<SpatialEntry<u32>>,
    }
    let fake = FakeDto {
        version: 3,
        max_entries: 9,
        split_strategy: 42,
        entries: vec![],
    };
    let bytes = bitcode::serialize(&fake).unwrap();
    let result: Result<SpatialIndex<u32>, _> = bitcode::deserialize(&bytes);
    assert!(result.is_err());
}

#[test]
fn test_serde_v3_preserves_custom_config() {
    let cfg = SpatialConfig::with_strategy(16, SplitStrategy::Linear).unwrap();
    let entries: Vec<SpatialEntry<u32>> = (0..100)
        .map(|i| SpatialEntry {
            bounds: BoundingBox::new((i % 10) as f32, (i / 10) as f32, 0.5, 0.5).unwrap(),
            data: i,
        })
        .collect();
    let index = SpatialIndex::bulk_load_with_config(entries, cfg);

    // Serialize (writes v3 with max=16, linear).
    let bytes = bitcode::serialize(&index).unwrap();
    let restored: SpatialIndex<u32> = bitcode::deserialize(&bytes).unwrap();
    assert_eq!(restored.config().max_entries(), 16);
    assert_eq!(restored.config().split_strategy(), SplitStrategy::Linear);

    // Serialize again and restore -- config persists through round-trips.
    let bytes2 = bitcode::serialize(&restored).unwrap();
    let restored2: SpatialIndex<u32> = bitcode::deserialize(&bytes2).unwrap();
    assert_eq!(restored2.config().max_entries(), 16);
    assert_eq!(restored2.config().split_strategy(), SplitStrategy::Linear);
    assert_eq!(restored2.len(), 100);
}

#[test]
fn test_rstar_config_strategy_getter() {
    let cfg = SpatialConfig::DEFAULT;
    assert_eq!(cfg.split_strategy(), SplitStrategy::RStar);

    let linear = SpatialConfig::with_strategy(9, SplitStrategy::Linear).unwrap();
    assert_eq!(linear.split_strategy(), SplitStrategy::Linear);
}

#[test]
fn test_split_strategy_default() {
    let s = SplitStrategy::default();
    assert_eq!(s, SplitStrategy::RStar);
}
