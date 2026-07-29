"""MeshLab/VCGLib filter kernels exposed through a flat C ABI."""

from std.algorithm import parallelize
from std.gpu.host import DeviceContext
from std.math import abs, acos, cos, floor, sin, sqrt
from std.sys import simd_width_of

comptime F64Ptr = UnsafePointer[Float64, AnyOrigin[mut=True]]
comptime I64Ptr = UnsafePointer[Int64, AnyOrigin[mut=True]]
comptime U8Ptr = UnsafePointer[UInt8, AnyOrigin[mut=True]]


@always_inline
def fp(address: Int) -> F64Ptr:
    return F64Ptr(unsafe_from_address=address)


@always_inline
def ip(address: Int) -> I64Ptr:
    return I64Ptr(unsafe_from_address=address)


@always_inline
def up(address: Int) -> U8Ptr:
    return U8Ptr(unsafe_from_address=address)


@always_inline
def infinity() -> Float64:
    var zero: Float64 = 0.0
    return 1.0 / zero


@always_inline
def edge_length(v: F64Ptr, a: Int, b: Int) -> Float64:
    var dx = v[3 * a] - v[3 * b]
    var dy = v[3 * a + 1] - v[3 * b + 1]
    var dz = v[3 * a + 2] - v[3 * b + 2]
    return sqrt(dx * dx + dy * dy + dz * dz)


# vcglib: vcg/complex/algorithms/stat.h Stat::ComputeMeshArea,
# ComputeShellBarycenter; vcg/complex/algorithms/inertia.h Inertia::Compute
@always_inline
def geometric_face_range(
    v: F64Ptr,
    f: I64Ptr,
    begin: Int,
    end: Int,
    sums: F64Ptr,
):
    var area_sum: Float64 = 0.0
    var shell_x: Float64 = 0.0
    var shell_y: Float64 = 0.0
    var shell_z: Float64 = 0.0
    var volume6: Float64 = 0.0
    for face in range(begin, end):
        var a = Int(f[3 * face])
        var b = Int(f[3 * face + 1])
        var c = Int(f[3 * face + 2])
        var ax = v[3 * a]
        var ay = v[3 * a + 1]
        var az = v[3 * a + 2]
        var bx = v[3 * b]
        var by = v[3 * b + 1]
        var bz = v[3 * b + 2]
        var cx = v[3 * c]
        var cy = v[3 * c + 1]
        var cz = v[3 * c + 2]
        var ux = bx - ax
        var uy = by - ay
        var uz = bz - az
        var wx = cx - ax
        var wy = cy - ay
        var wz = cz - az
        var nx = uy * wz - uz * wy
        var ny = uz * wx - ux * wz
        var nz = ux * wy - uy * wx
        var area = 0.5 * sqrt(nx * nx + ny * ny + nz * nz)
        area_sum += area
        shell_x += area * (ax + bx + cx) / 3.0
        shell_y += area * (ay + by + cy) / 3.0
        shell_z += area * (az + bz + cz) / 3.0
        volume6 += (
            ax * (by * cz - bz * cy)
            + ay * (bz * cx - bx * cz)
            + az * (bx * cy - by * cx)
        )
    sums[0] = area_sum
    sums[1] = shell_x
    sums[2] = shell_y
    sums[3] = shell_z
    sums[4] = volume6


@export("mml_geometric_measures")
def geometric_measures(
    vertices_address: Int,
    faces_address: Int,
    faux_address: Int,
    vertex_count: Int,
    face_count: Int,
    result_address: Int,
    keys_address: Int,
    states_address: Int,
    table_size: Int,
    scratch_address: Int,
    worker_count: Int,
) abi("C"):
    var v = fp(vertices_address)
    var f = ip(faces_address)
    var faux = up(faux_address)
    var r = fp(result_address)
    var keys = ip(keys_address)
    var states = up(states_address)
    var scratch = fp(scratch_address)
    for i in range(13):
        r[i] = 0.0

    if vertex_count > 0:
        comptime W = simd_width_of[DType.float64]()
        var sum_x = SIMD[DType.float64, W](0.0)
        var sum_y = SIMD[DType.float64, W](0.0)
        var sum_z = SIMD[DType.float64, W](0.0)
        var i = 0
        while i + W <= vertex_count:
            sum_x += (v + 3 * i).strided_load[width=W](3)
            sum_y += (v + 3 * i + 1).strided_load[width=W](3)
            sum_z += (v + 3 * i + 2).strided_load[width=W](3)
            i += W
        r[7] = sum_x.reduce_add()
        r[8] = sum_y.reduce_add()
        r[9] = sum_z.reduce_add()
        while i < vertex_count:
            r[7] += v[3 * i]
            r[8] += v[3 * i + 1]
            r[9] += v[3 * i + 2]
            i += 1
        r[7] /= Float64(vertex_count)
        r[8] /= Float64(vertex_count)
        r[9] /= Float64(vertex_count)

    if face_count > 0:
        var active_workers = worker_count
        if active_workers > 1:
            @parameter
            def work(worker: Int):
                var begin = worker * face_count // active_workers
                var end = (worker + 1) * face_count // active_workers
                geometric_face_range(
                    fp(vertices_address),
                    ip(faces_address),
                    begin,
                    end,
                    fp(scratch_address) + 5 * worker,
                )

            try:
                var cpu_context = DeviceContext(api="cpu")
                parallelize[work](worker_count, cpu_context)
            except:
                geometric_face_range(v, f, 0, face_count, scratch)
                active_workers = 1
        else:
            geometric_face_range(v, f, 0, face_count, scratch)
        for worker in range(active_workers):
            r[0] += scratch[5 * worker]
            r[3] += scratch[5 * worker + 1]
            r[4] += scratch[5 * worker + 2]
            r[5] += scratch[5 * worker + 3]
            r[6] += scratch[5 * worker + 4]

    var area_sum = r[0]
    if area_sum > 0.0:
        r[3] /= area_sum
        r[4] /= area_sum
        r[5] /= area_sum
    r[6] = abs(r[6]) / 6.0

    comptime W = simd_width_of[DType.float64]()
    var empty_keys = SIMD[DType.int64, W](-1)
    var slot_index = 0
    while slot_index + W <= table_size:
        keys.store(slot_index, empty_keys)
        slot_index += W
    while slot_index < table_size:
        keys[slot_index] = -1
        slot_index += 1

    var unique_sum: Float64 = 0.0
    var faux_sum: Float64 = 0.0
    var unique_count = 0
    var faux_count = 0
    var not_two_count = 0
    for face in range(face_count):
        var id0 = Int(f[3 * face])
        var id1 = Int(f[3 * face + 1])
        var id2 = Int(f[3 * face + 2])
        for edge in range(3):
            var a = id0 if edge == 0 else (id1 if edge == 1 else id2)
            var b = id1 if edge == 0 else (id2 if edge == 1 else id0)
            var lo = min(a, b)
            var hi = max(a, b)
            var key = Int64(lo) * Int64(vertex_count) + Int64(hi)
            var slot = edge_slot(lo, hi, table_size)
            while keys[slot] >= 0 and keys[slot] != key:
                slot = (slot + 1) % table_size
            var is_faux = faux[3 * face + edge] != 0
            if keys[slot] < 0:
                keys[slot] = key
                states[slot] = UInt8(5 if is_faux else 1)
                var length = edge_length(v, lo, hi)
                unique_sum += length
                unique_count += 1
                not_two_count += 1
                if is_faux:
                    faux_sum += length
                    faux_count += 1
            else:
                var state = Int(states[slot])
                var count = state & 3
                if count == 1:
                    count = 2
                    not_two_count -= 1
                elif count == 2:
                    count = 3
                    not_two_count += 1
                if is_faux and state & 4 == 0:
                    faux_sum += edge_length(v, lo, hi)
                    faux_count += 1
                    state |= 4
                states[slot] = UInt8(count | (state & 4))
    r[1] = unique_sum
    r[2] = Float64(unique_count)
    r[10] = faux_sum
    r[11] = Float64(faux_count)
    r[12] = 1.0 if not_two_count == 0 and face_count > 0 else 0.0


# vcglib: vcg/space/triangle3.h QualityRadii and DoubleArea
@export("mml_face_metrics")
def face_metrics(
    vertices_address: Int,
    faces_address: Int,
    face_count: Int,
    metric: Int,
    area_address: Int,
    quality_address: Int,
) abi("C"):
    var v = fp(vertices_address)
    var f = ip(faces_address)
    var areas = fp(area_address)
    var quality = fp(quality_address)
    for face in range(face_count):
        var ia = Int(f[3 * face])
        var ib = Int(f[3 * face + 1])
        var ic = Int(f[3 * face + 2])
        var a = edge_length(v, ia, ib)
        var b = edge_length(v, ia, ic)
        var c = edge_length(v, ib, ic)
        var s = (a + b + c) * 0.5
        var area2 = s * (a + b - s) * (a + c - s) * (b + c - s)
        areas[face] = sqrt(max(0.0, area2))
        if metric == 0:
            quality[face] = (
                2.0 * areas[face] / max(a * a, max(b * b, c * c))
                if areas[face] > 0.0
                else 0.0
            )
        elif metric == 1:
            quality[face] = (
                8.0 * area2 / (a * b * c * s)
                if area2 > 0.0 and a > 0.0 and b > 0.0 and c > 0.0 and s > 0.0
                else 0.0
            )
        elif metric == 2:
            quality[face] = (
                4.0 * sqrt(3.0) * areas[face] / (a * a + b * b + c * c)
                if areas[face] > 0.0
                else 0.0
            )
        else:
            quality[face] = areas[face]


# vcglib: vcg/space/color4.h Color4::SetColorRamp
@export("mml_color_ramp")
def color_ramp(
    scalar_address: Int,
    count: Int,
    minimum: Float64,
    maximum: Float64,
    colors_address: Int,
) abi("C"):
    var q = fp(scalar_address)
    var colors = fp(colors_address)
    var step = (maximum - minimum) / 4.0
    for i in range(count):
        var x = q[i]
        var red: Float64 = 0.0
        var green: Float64 = 0.0
        var blue: Float64 = 0.0
        if step <= 0.0 or x < minimum:
            red = 1.0
        else:
            x -= minimum
            if x < step:
                red = 1.0
                green = x / step
            elif x < 2.0 * step:
                red = 2.0 - x / step
                green = 1.0
            elif x < 3.0 * step:
                green = 1.0
                blue = x / step - 2.0
            elif x < 4.0 * step:
                green = 4.0 - x / step
                blue = 1.0
            else:
                blue = 1.0
        colors[4 * i] = red
        colors[4 * i + 1] = green
        colors[4 * i + 2] = blue
        colors[4 * i + 3] = 1.0


# vcglib: vcg/complex/algorithms/update/selection.h FaceOutOfRangeEdge
@export("mml_select_long_edges")
def select_long_edges(
    vertices_address: Int,
    faces_address: Int,
    face_count: Int,
    threshold: Float64,
    selected_address: Int,
) abi("C") -> Int:
    var v = fp(vertices_address)
    var f = ip(faces_address)
    var selected = ip(selected_address)
    var count = 0
    for face in range(face_count):
        var a = Int(f[3 * face])
        var b = Int(f[3 * face + 1])
        var c = Int(f[3 * face + 2])
        var hit = (
            edge_length(v, a, b) >= threshold
            or edge_length(v, b, c) >= threshold
            or edge_length(v, c, a) >= threshold
        )
        selected[face] = 1 if hit else 0
        count += 1 if hit else 0
    return count


# vcglib: vcg/complex/algorithms/update/selection.h VertexFromQualityRange
@export("mml_select_scalar")
def select_scalar(
    scalar_address: Int,
    count: Int,
    minimum: Float64,
    maximum: Float64,
    selected_address: Int,
) abi("C") -> Int:
    var q = fp(scalar_address)
    var selected = ip(selected_address)
    var total = 0
    for i in range(count):
        var hit = q[i] >= minimum and q[i] <= maximum
        selected[i] = 1 if hit else 0
        total += 1 if hit else 0
    return total


# vcglib: vcg/complex/algorithms/clean.h RemoveFaceOutOfRangeArea
@export("mml_mark_null_faces")
def mark_null_faces(
    vertices_address: Int,
    faces_address: Int,
    face_count: Int,
    keep_address: Int,
) abi("C") -> Int:
    var v = fp(vertices_address)
    var f = ip(faces_address)
    var keep = ip(keep_address)
    var removed = 0
    for face in range(face_count):
        var a = Int(f[3 * face])
        var b = Int(f[3 * face + 1])
        var c = Int(f[3 * face + 2])
        var ux = v[3 * b] - v[3 * a]
        var uy = v[3 * b + 1] - v[3 * a + 1]
        var uz = v[3 * b + 2] - v[3 * a + 2]
        var wx = v[3 * c] - v[3 * a]
        var wy = v[3 * c + 1] - v[3 * a + 1]
        var wz = v[3 * c + 2] - v[3 * a + 2]
        var nx = uy * wz - uz * wy
        var ny = uz * wx - ux * wz
        var nz = ux * wy - uy * wx
        var hit = nx * nx + ny * ny + nz * nz <= 0.0
        keep[face] = 0 if hit else 1
        removed += 1 if hit else 0
    return removed


# vcglib: vcg/complex/algorithms/clean.h RemoveDuplicateVertex
@export("mml_duplicate_vertex_remap")
def duplicate_vertex_remap(
    vertices_address: Int,
    vertex_count: Int,
    remap_address: Int,
) abi("C") -> Int:
    var v = fp(vertices_address)
    var remap = ip(remap_address)
    var unique_count = 0
    for i in range(vertex_count):
        var found = -1
        for j in range(i):
            if (
                v[3 * i] == v[3 * j]
                and v[3 * i + 1] == v[3 * j + 1]
                and v[3 * i + 2] == v[3 * j + 2]
            ):
                found = Int(remap[j])
                break
        if found >= 0:
            remap[i] = Int64(found)
        else:
            remap[i] = Int64(unique_count)
            unique_count += 1
    return vertex_count - unique_count


# vcglib: vcg/complex/algorithms/clean.h RemoveDuplicateFace SortedTriple
@export("mml_mark_duplicate_faces")
def mark_duplicate_faces(
    faces_address: Int,
    face_count: Int,
    keep_address: Int,
) abi("C") -> Int:
    var f = ip(faces_address)
    var keep = ip(keep_address)
    var removed = 0
    for i in range(face_count):
        var a = Int(f[3 * i])
        var b = Int(f[3 * i + 1])
        var c = Int(f[3 * i + 2])
        var lo = min(a, min(b, c))
        var hi = max(a, max(b, c))
        var mid = a + b + c - lo - hi
        var duplicate = False
        for j in range(i):
            if keep[j] == 0:
                continue
            var x = Int(f[3 * j])
            var y = Int(f[3 * j + 1])
            var z = Int(f[3 * j + 2])
            var lo2 = min(x, min(y, z))
            var hi2 = max(x, max(y, z))
            var mid2 = x + y + z - lo2 - hi2
            if lo == lo2 and mid == mid2 and hi == hi2:
                duplicate = True
                break
        keep[i] = 0 if duplicate else 1
        removed += 1 if duplicate else 0
    return removed


@always_inline
def edge_slot(a0: Int, b0: Int, size: Int) -> Int:
    var a = min(a0, b0)
    var b = max(a0, b0)
    return (a * 73856093 + b * 19349663) % size


# vcglib: vcg/complex/algorithms/refine.h Refine with MidPoint
@export("mml_midpoint_subdivide")
def midpoint_subdivide(
    vertices_address: Int,
    faces_address: Int,
    vertex_count: Int,
    face_count: Int,
    new_vertices_address: Int,
    new_faces_address: Int,
    table_a_address: Int,
    table_b_address: Int,
    table_value_address: Int,
    table_size: Int,
) abi("C") -> Int:
    var v = fp(vertices_address)
    var f = ip(faces_address)
    var nv = fp(new_vertices_address)
    var nf = ip(new_faces_address)
    var ta = ip(table_a_address)
    var tb = ip(table_b_address)
    var tv = ip(table_value_address)
    for i in range(table_size):
        ta[i] = -1
        tb[i] = -1
        tv[i] = -1
    for i in range(3 * vertex_count):
        nv[i] = v[i]
    var next_vertex = vertex_count
    for face in range(face_count):
        var id0 = Int(f[3 * face])
        var id1 = Int(f[3 * face + 1])
        var id2 = Int(f[3 * face + 2])
        var mid0 = 0
        var mid1 = 0
        var mid2 = 0
        for edge in range(3):
            var a = id0 if edge == 0 else (id1 if edge == 1 else id2)
            var b = id1 if edge == 0 else (id2 if edge == 1 else id0)
            var lo = min(a, b)
            var hi = max(a, b)
            var slot = edge_slot(lo, hi, table_size)
            while ta[slot] >= 0:
                if Int(ta[slot]) == lo and Int(tb[slot]) == hi:
                    break
                slot = (slot + 1) % table_size
            if ta[slot] < 0:
                ta[slot] = Int64(lo)
                tb[slot] = Int64(hi)
                tv[slot] = Int64(next_vertex)
                nv[3 * next_vertex] = (v[3 * lo] + v[3 * hi]) * 0.5
                nv[3 * next_vertex + 1] = (v[3 * lo + 1] + v[3 * hi + 1]) * 0.5
                nv[3 * next_vertex + 2] = (v[3 * lo + 2] + v[3 * hi + 2]) * 0.5
                next_vertex += 1
            if edge == 0:
                mid0 = Int(tv[slot])
            elif edge == 1:
                mid1 = Int(tv[slot])
            else:
                mid2 = Int(tv[slot])
        var base = 12 * face
        nf[base] = Int64(mid0)
        nf[base + 1] = Int64(mid1)
        nf[base + 2] = Int64(mid2)
        nf[base + 3] = Int64(id0)
        nf[base + 4] = Int64(mid0)
        nf[base + 5] = Int64(mid2)
        nf[base + 6] = Int64(id1)
        nf[base + 7] = Int64(mid1)
        nf[base + 8] = Int64(mid0)
        nf[base + 9] = Int64(id2)
        nf[base + 10] = Int64(mid2)
        nf[base + 11] = Int64(mid1)
    return next_vertex


@always_inline
def unfolded_distance(
    v: F64Ptr,
    point: Int,
    wing: Int,
    current: Int,
    wing_distance: Float64,
    current_distance: Float64,
) -> Float64:
    var ew_c = edge_length(v, point, current)
    var ew_w = edge_length(v, point, wing)
    var ec_w = edge_length(v, wing, current)
    if (
        ew_c <= 0.0
        or ew_w <= 0.0
        or ec_w <= 0.0
        or wing_distance <= 0.0
        or current_distance <= 0.0
    ):
        return current_distance + ew_c
    var wcx = (v[3 * point] - v[3 * current]) / ew_c
    var wcy = (v[3 * point + 1] - v[3 * current + 1]) / ew_c
    var wcz = (v[3 * point + 2] - v[3 * current + 2]) / ew_c
    var wwx = (v[3 * point] - v[3 * wing]) / ew_w
    var wwy = (v[3 * point + 1] - v[3 * wing + 1]) / ew_w
    var wwz = (v[3 * point + 2] - v[3 * wing + 2]) / ew_w
    var wccx = (v[3 * wing] - v[3 * current]) / ec_w
    var wccy = (v[3 * wing + 1] - v[3 * current + 1]) / ec_w
    var wccz = (v[3 * wing + 2] - v[3 * current + 2]) / ec_w
    var alpha = acos(max(-1.0, min(1.0, wcx * wccx + wcy * wccy + wcz * wccz)))
    var s = (current_distance + wing_distance + ec_w) * 0.5
    var aa = s / ec_w
    var bb = aa * s
    var alpha_arg = max(0.0, (bb - aa * wing_distance) / current_distance)
    var alpha_prime = 2.0 * acos(min(1.0, sqrt(alpha_arg)))
    if alpha + alpha_prime > 3.141592653589793:
        return current_distance + ew_c
    var beta_arg = max(0.0, (bb - aa * current_distance) / wing_distance)
    var beta_prime = 2.0 * acos(min(1.0, sqrt(beta_arg)))
    var beta = acos(
        max(-1.0, min(1.0, -(wwx * wccx + wwy * wccy + wwz * wccz)))
    )
    if beta + beta_prime > 3.141592653589793:
        return wing_distance + ew_w
    var theta = 3.141592653589793 - alpha - alpha_prime
    var delta = cos(theta) * ew_c
    var height = sin(theta) * ew_c
    return sqrt(height * height + (current_distance + delta) * (current_distance + delta))


# vcglib: vcg/complex/algorithms/geodesic.h Geodesic::Visit/GeoDistance
@export("mml_geodesic")
def geodesic(
    vertices_address: Int,
    faces_address: Int,
    offsets_address: Int,
    incident_faces_address: Int,
    vertex_count: Int,
    seed_address: Int,
    seed_count: Int,
    distances_address: Int,
    visited_address: Int,
    sources_address: Int,
) abi("C"):
    var v = fp(vertices_address)
    var faces = ip(faces_address)
    var offsets = ip(offsets_address)
    var incidents = ip(incident_faces_address)
    var seeds = ip(seed_address)
    var distances = fp(distances_address)
    var visited = ip(visited_address)
    var sources = ip(sources_address)
    for i in range(vertex_count):
        distances[i] = infinity()
        visited[i] = 0
        sources[i] = -1
    for i in range(seed_count):
        var seed = Int(seeds[i])
        distances[seed] = 0.0
        sources[seed] = Int64(seed)
    for _ in range(vertex_count):
        var current = -1
        var best = infinity()
        for i in range(vertex_count):
            if visited[i] == 0 and distances[i] < best:
                best = distances[i]
                current = i
        if current < 0:
            break
        visited[current] = 1
        for pos in range(Int(offsets[current]), Int(offsets[current + 1])):
            var face = Int(incidents[pos])
            var a = Int(faces[3 * face])
            var b = Int(faces[3 * face + 1])
            var c = Int(faces[3 * face + 2])
            var first = b if current == a else (c if current == b else a)
            var second = c if current == a else (a if current == b else b)
            for side in range(2):
                var point = first if side == 0 else second
                var wing = second if side == 0 else first
                var inter = edge_length(v, current, wing)
                var wing_distance = distances[wing]
                var tolerance = (inter + best + wing_distance) * 0.0001
                var candidate: Float64
                if (
                    sources[wing] != sources[current]
                    or inter + best < wing_distance + tolerance
                    or inter + wing_distance < best + tolerance
                    or best + wing_distance < inter + tolerance
                ):
                    candidate = best + edge_length(v, point, current)
                else:
                    candidate = unfolded_distance(
                        v, point, wing, current, wing_distance, best
                    )
                if candidate < distances[point]:
                    distances[point] = candidate
                    sources[point] = sources[current]


# vcglib: vcg/complex/algorithms/point_sampling.h SurfaceSampling::Montecarlo
@export("mml_sample_barycentric")
def sample_barycentric(
    vertices_address: Int,
    faces_address: Int,
    face_indices_address: Int,
    barycentric_address: Int,
    sample_count: Int,
    samples_address: Int,
) abi("C"):
    var v = fp(vertices_address)
    var f = ip(faces_address)
    var fi = ip(face_indices_address)
    var bary = fp(barycentric_address)
    var samples = fp(samples_address)
    for i in range(sample_count):
        var face = Int(fi[i])
        var a = Int(f[3 * face])
        var b = Int(f[3 * face + 1])
        var c = Int(f[3 * face + 2])
        var u = bary[2 * i]
        var w = bary[2 * i + 1]
        var su = sqrt(u)
        var wa = 1.0 - su
        var wb = su * (1.0 - w)
        var wc = su * w
        for axis in range(3):
            samples[3 * i + axis] = (
                wa * v[3 * a + axis]
                + wb * v[3 * b + axis]
                + wc * v[3 * c + axis]
            )


# vcglib: vcg/complex/algorithms/smooth.h Smooth::VertexCoordLaplacian
@export("mml_laplacian_step")
def laplacian_step(
    vertices_address: Int,
    offsets_address: Int,
    neighbors_address: Int,
    extra_self_address: Int,
    vertex_count: Int,
    relaxation: Float64,
    result_address: Int,
) abi("C"):
    var v = fp(vertices_address)
    var offsets = ip(offsets_address)
    var neighbors = ip(neighbors_address)
    var extra_self = ip(extra_self_address)
    var result = fp(result_address)
    for i in range(vertex_count):
        var begin = Int(offsets[i])
        var end = Int(offsets[i + 1])
        for axis in range(3):
            if begin == end:
                result[3 * i + axis] = v[3 * i + axis]
            else:
                var mean = v[3 * i + axis] * Float64(1 + extra_self[i])
                for pos in range(begin, end):
                    mean += v[3 * Int(neighbors[pos]) + axis]
                mean /= Float64(end - begin + 1 + Int(extra_self[i]))
                result[3 * i + axis] = (
                    (1.0 - relaxation) * v[3 * i + axis] + relaxation * mean
                )
