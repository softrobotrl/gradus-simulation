FeatureScript 3083;
import(path : "onshape/std/geometry.fs", version : "3083.0");

annotation { "Feature Type Name" : "Gradus Dog", "Feature Type Description" : "Simple quadruped concept: planar two-link legs, two antagonistic tendons per leg (8 total)." }
export const gradusDog = defineFeature(function(context is Context, id is Id, definition is map)
    precondition
    {
    }
    {
        const bodyCol = color(0.82, 0.83, 0.85);
        const legCol = color(0.95, 0.58, 0.12);
        const darkCol = color(0.16, 0.17, 0.19);
        const metalCol = color(0.55, 0.57, 0.60);
        const redCol = color(0.85, 0.12, 0.12);
        const blueCol = color(0.12, 0.35, 0.85);

        var bodyParts = [];

        // Body, head, and shell.
        bodyParts = append(bodyParts, addBox(context, id + "body", vector(0, 0, 215) * millimeter, vector(300, 160, 80) * millimeter, bodyCol));
        bodyParts = append(bodyParts, addBox(context, id + "neck", vector(155, 0, 230) * millimeter, vector(50, 80, 55) * millimeter, bodyCol));
        bodyParts = append(bodyParts, addBox(context, id + "head", vector(205, 0, 240) * millimeter, vector(95, 95, 65) * millimeter, bodyCol));
        bodyParts = append(bodyParts, addBox(context, id + "snout", vector(250, 0, 215) * millimeter, vector(55, 65, 35) * millimeter, bodyCol));
        bodyParts = append(bodyParts, addBox(context, id + "earL", vector(185, 50, 285) * millimeter, vector(25, 16, 50) * millimeter, legCol));
        bodyParts = append(bodyParts, addBox(context, id + "earR", vector(185, -50, 285) * millimeter, vector(25, 16, 50) * millimeter, legCol));
        bodyParts = append(bodyParts, addBox(context, id + "tail", vector(-170, 0, 235) * millimeter, vector(50, 20, 18) * millimeter, legCol));
        bodyParts = append(bodyParts, addBox(context, id + "tailTip", vector(-205, 0, 245) * millimeter, vector(30, 16, 16) * millimeter, legCol));
        bodyParts = append(bodyParts, addBox(context, id + "topPlate", vector(0, 0, 258) * millimeter, vector(220, 120, 12) * millimeter, darkCol));
        bodyParts = append(bodyParts, addBox(context, id + "battery", vector(0, 0, 272) * millimeter, vector(150, 95, 20) * millimeter, darkCol));
        bodyParts = append(bodyParts, addSphere(context, id + "eyeL", vector(250, 25, 250) * millimeter, 8 * millimeter, darkCol));
        bodyParts = append(bodyParts, addSphere(context, id + "eyeR", vector(250, -25, 250) * millimeter, 8 * millimeter, darkCol));
        bodyParts = append(bodyParts, addCylinder(context, id + "camera", vector(270, 0, 228) * millimeter, vector(285, 0, 228) * millimeter, 7 * millimeter, darkCol));
        bodyParts = append(bodyParts, addSphere(context, id + "lens", vector(285, 0, 228) * millimeter, 5 * millimeter, blueCol));
        bodyParts = append(bodyParts, addSphere(context, id + "nose", vector(282, 0, 208) * millimeter, 7 * millimeter, darkCol));

        // Four planar legs. Left/right are mirrored; front/rear knee directions are mirrored.
        const flBody = addLeg(context, id + "FL", vector(100, 105, 175) * millimeter, vector(140, 105, 115) * millimeter, vector(110, 105, 15) * millimeter, 1, true, legCol, darkCol, metalCol, redCol, blueCol);
        const frBody = addLeg(context, id + "FR", vector(100, -105, 175) * millimeter, vector(140, -105, 115) * millimeter, vector(110, -105, 15) * millimeter, -1, true, legCol, darkCol, metalCol, redCol, blueCol);
        const rlBody = addLeg(context, id + "RL", vector(-100, 105, 175) * millimeter, vector(-140, 105, 115) * millimeter, vector(-110, 105, 15) * millimeter, 1, false, legCol, darkCol, metalCol, redCol, blueCol);
        const rrBody = addLeg(context, id + "RR", vector(-100, -105, 175) * millimeter, vector(-140, -105, 115) * millimeter, vector(-110, -105, 15) * millimeter, -1, false, legCol, darkCol, metalCol, redCol, blueCol);

        mergeBodies(context, id + "bodyMerge", concatenateArrays([bodyParts, flBody, frBody, rlBody, rrBody]), bodyCol);
    });

function addBox(context is Context, id is Id, center is Vector, size is Vector, col is Color) returns Query
{
    const half = size / 2;
    fCuboid(context, id, { "corner1" : center - half, "corner2" : center + half });
    const body = qCreatedBy(id, EntityType.BODY);
    setProperty(context, { "entities" : body, "propertyType" : PropertyType.APPEARANCE, "value" : col });
    return body;
}

function addCylinder(context is Context, id is Id, a is Vector, b is Vector, radius is ValueWithUnits, col is Color) returns Query
{
    fCylinder(context, id, { "topCenter" : a, "bottomCenter" : b, "radius" : radius });
    const body = qCreatedBy(id, EntityType.BODY);
    setProperty(context, { "entities" : body, "propertyType" : PropertyType.APPEARANCE, "value" : col });
    return body;
}

function addSphere(context is Context, id is Id, center is Vector, radius is ValueWithUnits, col is Color) returns Query
{
    opSphere(context, id, { "center" : center, "radius" : radius });
    const body = qCreatedBy(id, EntityType.BODY);
    setProperty(context, { "entities" : body, "propertyType" : PropertyType.APPEARANCE, "value" : col });
    return body;
}

function mergeBodies(context is Context, id is Id, bodies is array, col is Color) returns Query
{
    if (size(bodies) == 0)
        throw regenError("No bodies to merge");
    opBoolean(context, id, { "tools" : qUnion(bodies), "operationType" : BooleanOperationType.UNION });
    // UNION preserves the identity/color of the first tool body.
    return bodies[0];
}

function addLeg(context is Context, id is Id, hip is Vector, knee is Vector, foot is Vector, side is number, front is boolean, legCol is Color, darkCol is Color, metalCol is Color, redCol is Color, blueCol is Color) returns array
{
    const x = hip[0];
    const y = hip[1];
    const z = hip[2];

    // Front legs take the tendon guide offsets in +x, rear legs mirror them.
    const xSign = front ? 1 : -1;

    var bodyParts = [];
    var thighParts = [];
    var calfParts = [];

    // Two-link leg in the X-Z plane.
    thighParts = append(thighParts, addCylinder(context, id + "thigh", hip, knee, 10 * millimeter, legCol));
    thighParts = append(thighParts, addSphere(context, id + "hipJoint", hip, 13 * millimeter, darkCol));
    thighParts = append(thighParts, addSphere(context, id + "kneeJoint", knee, 10 * millimeter, darkCol));
    calfParts = append(calfParts, addCylinder(context, id + "calf", knee, foot, 8 * millimeter, legCol));
    calfParts = append(calfParts, addSphere(context, id + "foot", foot, 15 * millimeter, darkCol));

    // Hip fork keeps the leg planar.
    bodyParts = append(bodyParts, addBox(context, id + "forkInner", vector(x, y - side * 20.5 * millimeter, z - 5 * millimeter), vector(50, 5, 50) * millimeter, darkCol));
    bodyParts = append(bodyParts, addBox(context, id + "forkOuter", vector(x, y + side * 20.5 * millimeter, z - 5 * millimeter), vector(50, 5, 50) * millimeter, darkCol));
    bodyParts = append(bodyParts, addBox(context, id + "forkTop", vector(x, y, z + 20 * millimeter), vector(50, 56, 10) * millimeter, darkCol));
    bodyParts = append(bodyParts, addCylinder(context, id + "hipBolt", vector(x, y - side * 28 * millimeter, z), vector(x, y + side * 28 * millimeter, z), 5 * millimeter, metalCol));

    // Two tendon drums per leg. The hip is a passive pivot; the two tendons
    // are the only leg actuators (2 per leg, 8 total).  The plate carries the
    // two drums 90 mm apart, one on each side of the hip.
    bodyParts = append(bodyParts, addBox(context, id + "drumPlate", vector(x, y + side * 22.5 * millimeter, z + 44 * millimeter), vector(118, 5, 40) * millimeter, darkCol));

    for (var i = 0; i < 2; i += 1)
    {
        // Per-tendon guide offsets in x: hip guide (= drum), knee guide, foot anchor.
        // The two tendons must not pass through both joints at the same time, or
        // both produce the same joint-torque direction and the leg cannot hold
        // itself up.  Tendon 0 runs past the knee (hip moment only), tendon 1 is
        // anchored beside it (knee moment with the opposite hip moment).
        //
        // The offset is the moment arm of the tendon: the joint torque is the
        // cable tension times the offset.  30 mm was enough to stand but the
        // knee tendon saturated while walking and the leg buckled, so the
        // guides sit 45 mm out.  These numbers match the simulation
        // (TENDON_GUIDES in pipeline/dog_model.py).
        const hipOffsetMm = (i == 0) ? 45 : -45;
        const kneeOffsetMm = (i == 0) ? 0 : 45;
        const footOffsetMm = (i == 0) ? 45 : -45;
        const hipOffset = hipOffsetMm * millimeter;
        const kneeOffset = kneeOffsetMm * millimeter;
        const footOffset = footOffsetMm * millimeter;
        const tendonCol = (i == 0) ? redCol : blueCol;
        const drumX = x + xSign * hipOffset;
        const drumY = y + side * 30 * millimeter;
        const drumZ = z + 44 * millimeter;
        const drumCenter = vector(drumX, drumY, drumZ);
        const hipGuide = vector(drumX, y, z);

        // Guide bar at the hip.
        bodyParts = append(bodyParts, addCylinder(context, id + ("hipGuide" ~ i), vector(drumX, y - side * 22 * millimeter, z), vector(drumX, y + side * 22 * millimeter, z), 4 * millimeter, metalCol));

        // Drum (12 mm radius) + axle, one body per tendon.
        var drumParts = [];
        drumParts = append(drumParts, addCylinder(context, id + ("drum" ~ i), vector(drumX, drumY - side * 7 * millimeter, drumZ), vector(drumX, drumY + side * 7 * millimeter, drumZ), 12 * millimeter, tendonCol));
        drumParts = append(drumParts, addCylinder(context, id + ("drumAxle" ~ i), vector(drumX, y + side * 22 * millimeter, drumZ), vector(drumX, y + side * 46 * millimeter, drumZ), 2 * millimeter, metalCol));
        mergeBodies(context, id + ("drumMerge" ~ i), drumParts, tendonCol);

        // Gearbox can behind the drum.  The winch reduces 2:1, so the motor has
        // to deliver 0.25 N*m to get 0.5 N*m at the drum: that is what keeps
        // the knee tendon from saturating while the dog walks.
        bodyParts = append(bodyParts, addCylinder(context, id + ("gearbox" ~ i), vector(drumX, y + side * 46 * millimeter, drumZ), vector(drumX, y + side * 60 * millimeter, drumZ), 11.5 * millimeter, metalCol));

        // Knee guide and foot anchor.
        const p2 = vector(knee[0] + xSign * kneeOffset, knee[1], knee[2]);
        const p3 = vector(foot[0] + xSign * footOffset, foot[1], foot[2]);
        const bracketCenter = vector((knee[0] + p2[0]) / 2, knee[1], knee[2]);
        const bracketLength = kneeOffsetMm / 2 + 14;
        thighParts = append(thighParts, addBox(context, id + ("kneeBracket" ~ i), bracketCenter, vector(bracketLength, 10, 10) * millimeter, darkCol));
        thighParts = append(thighParts, addCylinder(context, id + ("kneeGuide" ~ i), vector(p2[0], p2[1] - side * 8 * millimeter, p2[2]), vector(p2[0], p2[1] + side * 8 * millimeter, p2[2]), 3 * millimeter, metalCol));
        calfParts = append(calfParts, addBox(context, id + ("anchor" ~ i), p3, vector(12, 12, 12) * millimeter, darkCol));
        // Arm that carries the anchor beside the foot.
        calfParts = append(calfParts, addCylinder(context, id + ("anchorArm" ~ i), foot, p3, 4 * millimeter, darkCol));

        // One tendon: drum -> hip guide -> knee guide -> foot anchor.
        var tendonParts = [];
        tendonParts = append(tendonParts, addCylinder(context, id + ("tendonA" ~ i), drumCenter, hipGuide, 1.5 * millimeter, tendonCol));
        tendonParts = append(tendonParts, addSphere(context, id + ("tendonHip" ~ i), hipGuide, 2.5 * millimeter, tendonCol));
        tendonParts = append(tendonParts, addCylinder(context, id + ("tendonB" ~ i), hipGuide, p2, 1.5 * millimeter, tendonCol));
        tendonParts = append(tendonParts, addSphere(context, id + ("tendonKnee" ~ i), p2, 2 * millimeter, tendonCol));
        tendonParts = append(tendonParts, addCylinder(context, id + ("tendonC" ~ i), p2, p3, 1.5 * millimeter, tendonCol));
        mergeBodies(context, id + ("tendonMerge" ~ i), tendonParts, tendonCol);
    }

    mergeBodies(context, id + "thighMerge", thighParts, legCol);
    mergeBodies(context, id + "calfMerge", calfParts, legCol);
    return bodyParts;
}
