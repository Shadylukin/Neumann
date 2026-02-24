//! Integration tests for tensor_spatial public API.

use tensor_spatial::{
    BoundingBox, BoundingBox3D, BoundingBoxN, SpatialEntry, SpatialEntry3D, SpatialEntryN,
    SpatialError, SpatialIndex, SpatialIndex3D, SpatialIndexN,
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

    let orig_nearest = index.query_nearest(25.0, 12.0, 5);
    let rest_nearest = restored.query_nearest(25.0, 12.0, 5);
    let mut orig_data: Vec<_> = orig_nearest.iter().map(|e| e.data).collect();
    let mut rest_data: Vec<_> = rest_nearest.iter().map(|e| e.data).collect();
    orig_data.sort_unstable();
    rest_data.sort_unstable();
    assert_eq!(orig_data, rest_data);
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
        entries: Vec<SpatialEntry3D<u32>>,
    }
    let fake = FakeDto {
        version: 99,
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
        entries: Vec<SpatialEntry<u32>>,
    }
    let fake = FakeDto {
        version: 99,
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
